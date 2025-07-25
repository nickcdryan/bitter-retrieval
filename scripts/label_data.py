#!/usr/bin/env python3
"""
Data labeling script for MS MARCO dataset

This script generates soft labels (LLM losses) for MS MARCO data using a specified language model.

Usage:
    python scripts/label_data.py --model Qwen/Qwen3-8B-Base --split train --num 1000
    python scripts/label_data.py --model Qwen/Qwen3-8B-Base --all-splits --upload-hf "username/dataset-name"
    python scripts/label_data.py --model meta-llama/Llama-3.2-3B --split validation --num 0 --dtype bf16
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import os
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def compute_llm_soft_labels(data, llm, llm_tokenizer, device, batch_size, max_samples):
    """Add soft labels (LLM losses) to MS MARCO data"""
    
    # STEP 1: Collect all inputs across all examples
    all_inputs = []
    example_metadata = []
    
    for i, item in enumerate(tqdm(data, desc="Preparing inputs")):
        if max_samples > 0 and i >= max_samples:
            break

        query = item["query"]
        answers = item["answers"]
        passages = item["passages"]["passage_text"]
        passages.append("")  # add empty string
        is_selected = item["passages"]["is_selected"]
        is_selected.append(0)  # add empty string label

        # Skip if no answer
        if not answers:
            continue

        answer = answers[0]

        # Store metadata for this example
        example_metadata.append({
            "query": query,
            "answers": answers,
            "passages": passages,
            "is_selected": is_selected,
            "query_id": item.get("query_id", i),
            "query_type": item.get("query_type", ""),
            "wellFormedAnswers": item.get("wellFormedAnswers", []),
            "start_idx": len(all_inputs),  # Where this example's inputs start
            "num_passages": len(passages)
        })

        # Add all passages for this example to global list
        for passage in passages:
            input_text = f"Question: {query} Context: {passage} Answer: {answer}"
            # Pre-compute mask position during preparation
            target_start_idx = input_text.rfind("Answer: ") + len("Answer: ")
            target_start_token_idx = llm_tokenizer(input_text[:target_start_idx], return_tensors="pt")["input_ids"].shape[1] - 1
            all_inputs.append({
                "text": input_text,
                "mask_position": target_start_token_idx
            })

    print(f"Processing {len(all_inputs)} total inputs across {len(example_metadata)} examples")
    
    # STEP 2: Process all inputs in large batches
    all_losses = []
    
    for i in tqdm(range(0, len(all_inputs), batch_size), desc="Computing losses"):
        batch_data = all_inputs[i:i+batch_size]
        batch_texts = [item["text"] for item in batch_data]

        # Batch tokenize
        inputs = llm_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        labels = inputs["input_ids"].clone()

        # Use pre-computed mask positions
        for k, item in enumerate(batch_data):
            labels[k, :item["mask_position"]] = -100
        labels[labels==llm_tokenizer.pad_token_id] = -100

        # Get losses
        with torch.no_grad():
            outputs = llm(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logits = outputs.logits

            # Compute per-example losses
            for k in range(logits.shape[0]):
                example_logits = logits[k, :-1, :].contiguous()
                example_labels = labels[k, 1:].contiguous()
                valid_positions = example_labels != -100

                if valid_positions.sum() > 0:
                    loss_val = F.cross_entropy(example_logits, example_labels)
                    all_losses.append(loss_val.item())
                else:
                    all_losses.append(float('inf'))  # Invalid example

    # STEP 3: Reconstruct results by example
    soft_labeled_data = []
    
    for metadata in example_metadata:
        start_idx = metadata["start_idx"]
        num_passages = metadata["num_passages"]
        passage_losses = all_losses[start_idx:start_idx + num_passages]

        soft_labeled_item = {
            "query": metadata["query"],
            "answers": metadata["answers"],
            "passages": {
                "passage_text": metadata["passages"],
                "is_selected": metadata["is_selected"],
                "soft_labels": passage_losses
            },
            "query_id": metadata["query_id"],
            "query_type": metadata["query_type"],
            "wellFormedAnswers": metadata["wellFormedAnswers"]
        }

        soft_labeled_data.append(soft_labeled_item)

    return soft_labeled_data


def upload_to_hf_hub(data, dataset_name, split_name, model_name):
    """Upload data to Hugging Face Hub as a dataset"""
    try:
        from huggingface_hub import HfApi
        
        # Get HF token from environment
        hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
        if not hf_token:
            print("Error: No Hugging Face token found. Set HUGGINGFACE_TOKEN in .env file")
            return
        
        # Convert to HF Dataset
        dataset = Dataset.from_list(data)
        
        # Push to hub with explicit token
        dataset.push_to_hub(
            dataset_name,
            split=split_name,
            token=hf_token,
            commit_message=f"Add {split_name} split with {model_name} soft labels"
        )
        
        print(f"✅ Uploaded to https://huggingface.co/datasets/{dataset_name}")
        
    except ImportError:
        print("❌ Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        print(f"❌ HF Hub upload failed: {e}")


def process_all_splits(model_name, upload_hf_base, dtype="bf16", batch_size=64):
    """Process all splits (train, validation, test) with all examples"""
    splits = ["train", "validation", "test"]
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split with ALL examples...")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, __file__,
            "--model", model_name,
            "--split", split,
            "--num", "0",  # Process all examples
            "--dtype", dtype,
            "--batch-size", str(batch_size),
            "--upload-hf", upload_hf_base
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Completed {split} split successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to process {split} split: {e}")
            return False
    
    print(f"\n🎉 ALL SPLITS COMPLETED! Dataset: {upload_hf_base}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate soft labels for MS MARCO dataset")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-Base", help="HuggingFace model path")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="Dataset split")
    parser.add_argument("--start", type=int, default=0, help="Start point in dataset")
    parser.add_argument("--num", type=int, default=1000, help="Number of examples (use 0 or -1 for all examples in split)")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="LLM batch size")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bf16"], help="Model dtype (bf16/fp16 recommended for Flash Attention)")
    parser.add_argument("--upload-hf", help="Upload to HF Hub (dataset name, e.g. 'username/msmarco-soft-labels')")
    parser.add_argument("--all-splits", action="store_true", help="Process all splits (train, validation, test) with all examples")
    
    args = parser.parse_args()
    
    # Handle all-splits mode
    if args.all_splits:
        if not args.upload_hf:
            print("❌ Error: --upload-hf is required when using --all-splits")
            return
        
        success = process_all_splits(
            model_name=args.model,
            upload_hf_base=args.upload_hf,
            dtype=args.dtype,
            batch_size=args.batch_size
        )
        
        if not success:
            exit(1)
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load LLM
    print(f"🤖 Loading {args.model}...")
    
    # Set torch dtype based on argument
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Only use Flash Attention with fp16/bf16
    use_flash_attn = args.dtype in ["bf16", "float16"]
    
    try:
        if use_flash_attn:
            llm = AutoModelForCausalLM.from_pretrained(
                args.model, 
                attn_implementation="flash_attention_2", 
                torch_dtype=torch_dtype
            ).to(device).eval()
            print(f"✅ Loaded with Flash Attention 2 and {args.dtype}")
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                args.model, 
                torch_dtype=torch_dtype
            ).to(device).eval()
            print(f"✅ Loaded with {args.dtype} (Flash Attention disabled)")
    except Exception as e:
        print(f"⚠️  Failed to load with {args.dtype}, falling back to default: {e}")
        llm = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    
    llm_tokenizer = AutoTokenizer.from_pretrained(args.model)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    # Load dataset
    print("📂 Loading MS MARCO...")
    msmarco = load_dataset("microsoft/ms_marco", "v1.1")
    data = msmarco[args.split]
    
    # Handle "all examples" case
    if args.num <= 0:
        print(f"📊 Processing ALL examples in {args.split} split ({len(data)} total)")
        selected_data = data
        args.num = len(data)  # Update for downstream processing
    else:
        # Select range
        end_idx = min(args.start + args.num, len(data))
        print(f"📊 Processing examples {args.start} to {end_idx} ({end_idx - args.start} total)")
        selected_data = data.select(range(args.start, end_idx))
    
    # Generate soft labels
    print("🏷️  Generating soft labels...")
    soft_labeled_data = compute_llm_soft_labels(
        selected_data, llm, llm_tokenizer, device, args.batch_size, args.num
    )
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = args.model.replace("/", "_")
    filename = f"soft_labels_msmarco_{model_name}_{args.start}-{args.start + args.num}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(soft_labeled_data, f, indent=2)
    
    print(f"💾 Saved to {filepath}")
    
    # Upload to HF Hub if requested
    if args.upload_hf:
        model_name = args.model.split("/")[-1]  # Get just the model name
        print(f"📤 Uploading to HF Hub: {args.upload_hf}")
        upload_to_hf_hub(soft_labeled_data, args.upload_hf, args.split, model_name)
    
    print("✅ Data labeling completed successfully!")


if __name__ == "__main__":
    main() 