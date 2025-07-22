#!/usr/bin/env python3
"""
Clean MS MARCO soft labeling script with Hugging Face Hub upload

SETUP:
    poetry install torch transformers datasets huggingface_hub tqdm
    huggingface-cli login

TEST RUN (100 examples per split):
    python label_msmarco.py --split train --num 100 --upload-hf "nickcdryan/ms_marco_softlabel_Qwen3-4B"
    python label_msmarco.py --split validation --num 100 --upload-hf "nickcdryan/ms_marco_softlabel_Qwen3-4B"
    python label_msmarco.py --split test --num 100 --upload-hf "nickcdryan/ms_marco_softlabel_Qwen3-4B"

FULL RUN (all examples):
    python label_msmarco.py --split train --num 808731 --upload-hf "nickcdryan/ms_marco_softlabel_Qwen3-4B"
    python label_msmarco.py --split validation --num 101093 --upload-hf "nickcdryan/ms_marco_softlabel_Qwen3-4B"
    python label_msmarco.py --split test --num 101092 --upload-hf "nickcdryan/ms_marco_softlabel_Qwen3-4B"

DIFFERENT MODELS:
    # Llama 3.2 3B
    python label_msmarco.py --model "meta-llama/Llama-3.2-3B" --split train --num 100 --upload-hf "nickcdryan/ms_marco_softlabel_llama-3.2-3b"
    

NAMING CONVENTION:
    Dataset name format: "nickcdryan/ms_marco_softlabel_{model_name_simplified}"
    Examples:
    - Qwen/Qwen3-4B → nickcdryan/ms_marco_softlabel_Qwen3-4B
    - meta-llama/Llama-3.2-3B → nickcdryan/ms_marco_softlabel_llama-3.2-3b
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import argparse
from pathlib import Path


def compute_llm_soft_labels(data, llm, llm_tokenizer, device, batch_size, max_samples):
    """Add soft labels (LLM losses) to MS MARCO data"""
    soft_labeled_data = []

    for i, item in enumerate(tqdm(data, desc="Computing soft labels")):
        if i >= max_samples:
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

        # Create LLM inputs for all passages
        llm_inputs = []
        for passage in passages:
            input_text = f"Question: {query} Context: {passage} Answer: {answer}"
            llm_inputs.append(input_text)

        # Process through LLM in batches
        passage_losses = []

        for j in range(0, len(llm_inputs), batch_size):
            batch_inputs = llm_inputs[j:j+batch_size]

            # Batch tokenize
            inputs = llm_tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            labels = inputs["input_ids"].clone()

            # Mask everything before "Answer:"
            for k, text in enumerate(batch_inputs):
                target_start_idx = text.rfind("Answer: ") + len("Answer: ")
                target_start_token_idx = llm_tokenizer(text[:target_start_idx], return_tensors="pt")["input_ids"].shape[1] - 1
                labels[k, :target_start_token_idx] = -100
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

                    passage_losses.append(loss_val.item())

        # Create new item with soft labels
        soft_labeled_item = {
            "query": query,
            "answers": answers,
            "passages": {
                "passage_text": passages,
                "is_selected": is_selected,
                "soft_labels": passage_losses
            },
            "query_id": item.get("query_id", i),
            "query_type": item.get("query_type", ""),
            "wellFormedAnswers": item.get("wellFormedAnswers", [])
        }

        soft_labeled_data.append(soft_labeled_item)

    return soft_labeled_data


def upload_to_hf_hub(data, dataset_name, split_name, model_name):
    """Upload data to Hugging Face Hub as a dataset"""
    try:
        from huggingface_hub import HfApi
        
        # Convert to HF Dataset
        dataset = Dataset.from_list(data)
        
        # Push to hub
        dataset.push_to_hub(
            dataset_name,
            split=split_name,
            commit_message=f"Add {split_name} split with {model_name} soft labels"
        )
        
        print(f"Uploaded to https://huggingface.co/datasets/{dataset_name}")
        
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        print(f"HF Hub upload failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate soft labels for MS MARCO dataset")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model path")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="Dataset split")
    parser.add_argument("--start", type=int, default=0, help="Start point in dataset")
    parser.add_argument("--num", type=int, default=1000, help="Number of examples")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="LLM batch size")
    parser.add_argument("--upload-hf", help="Upload to HF Hub (dataset name, e.g. 'username/msmarco-soft-labels')")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load LLM
    print(f"Loading {args.model}...")
    llm = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    llm_tokenizer = AutoTokenizer.from_pretrained(args.model)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    # Load dataset
    print("Loading MS MARCO...")
    msmarco = load_dataset("microsoft/ms_marco", "v1.1")
    data = msmarco[args.split]
    
    # Select range
    selected_data = data.select(range(args.start, args.start + args.num))
    
    # Generate soft labels
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
    
    print(f"Saved to {filepath}")
    
    # Upload to HF Hub if requested
    if args.upload_hf:
        model_name = args.model.split("/")[-1]  # Get just the model name
        upload_to_hf_hub(soft_labeled_data, args.upload_hf, args.split, model_name)


if __name__ == "__main__":
    main() 