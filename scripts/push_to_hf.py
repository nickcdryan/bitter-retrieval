#!/usr/bin/env python3
"""
HuggingFace operations script for bitter retrieval

This script handles pushing trained models and datasets to HuggingFace Hub.

Usage:
    python scripts/push_to_hf.py --model models/my-model --repo username/model-name
    python scripts/push_to_hf.py --dataset data/my-dataset.json --repo username/dataset-name --type dataset
    python scripts/push_to_hf.py --list-local  # List available local models
"""

import argparse
import os
import json
from pathlib import Path
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("python-dotenv not installed - using system environment variables only")


def check_hf_token():
    """Check if HuggingFace token is available"""
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if not token:
        print("❌ Error: No HuggingFace token found")
        print("Set HUGGINGFACE_TOKEN in your .env file")
        return False
    return token


def push_model_to_hf(model_path, repo_name, private=False, commit_message=None):
    """Push a trained model to HuggingFace Hub"""
    try:
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ Required packages not installed. Run: pip install transformers huggingface_hub")
        return False
    
    token = check_hf_token()
    if not token:
        return False
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    print(f"🤖 Loading model from: {model_path}")
    
    try:
        # Load model and tokenizer
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        
        # Check if tokenizer exists
        tokenizer_path = model_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            has_tokenizer = True
        except:
            print("⚠️  No tokenizer found in model directory")
            has_tokenizer = False
        
        print(f"📤 Pushing model to: {repo_name}")
        
        # Push model
        model.push_to_hub(
            repo_name,
            token=token,
            private=private,
            commit_message=commit_message or f"Upload trained model from {model_path}"
        )
        
        # Push tokenizer if available
        if has_tokenizer:
            tokenizer.push_to_hub(
                repo_name,
                token=token,
                private=private,
                commit_message=commit_message or f"Upload tokenizer from {model_path}"
            )
        
        print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to push model: {e}")
        return False


def push_dataset_to_hf(dataset_path, repo_name, split_name="train", private=False, commit_message=None):
    """Push a dataset to HuggingFace Hub"""
    try:
        from datasets import Dataset
    except ImportError:
        print("❌ datasets package not installed. Run: pip install datasets")
        return False
    
    token = check_hf_token()
    if not token:
        return False
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    print(f"📊 Loading dataset from: {dataset_path}")
    
    try:
        # Load data based on file type
        if dataset_path.suffix == '.json':
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        else:
            print(f"❌ Unsupported file type: {dataset_path.suffix}")
            return False
        
        # Create HF Dataset
        dataset = Dataset.from_list(data)
        
        print(f"📤 Pushing dataset to: {repo_name} (split: {split_name})")
        
        # Push to hub
        dataset.push_to_hub(
            repo_name,
            split=split_name,
            token=token,
            private=private,
            commit_message=commit_message or f"Upload dataset from {dataset_path}"
        )
        
        print(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to push dataset: {e}")
        return False


def list_local_models():
    """List available local models"""
    print("🔍 Scanning for local models...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ No models directory found")
        return
    
    model_paths = []
    
    # Look for directories with model files
    for path in models_dir.rglob("*"):
        if path.is_dir():
            # Check if it contains model files
            has_model = any([
                (path / "pytorch_model.bin").exists(),
                (path / "model.safetensors").exists(),
                (path / "config.json").exists()
            ])
            if has_model:
                model_paths.append(path)
    
    if not model_paths:
        print("📭 No trained models found in models/ directory")
        return
    
    print(f"📚 Found {len(model_paths)} model(s):")
    for i, path in enumerate(model_paths, 1):
        # Get model info
        config_path = path / "config.json"
        model_info = ""
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    model_type = config.get("model_type", "unknown")
                    model_info = f" ({model_type})"
            except:
                pass
        
        # Check size
        size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        
        print(f"  {i}. {path}{model_info} - {size_mb:.1f} MB")


def create_model_card(repo_name, model_path, experiment_info=None):
    """Create a model card for the uploaded model"""
    model_card = f"""---
license: apache-2.0
tags:
- retrieval
- embedding
- bitter-retrieval
library_name: transformers
---

# {repo_name}

This is a retrieval model trained using the [Bitter Retrieval](https://github.com/user/bitter-retrieval) framework.

## Model Details

- **Model Type**: Dense Retrieval Encoder
- **Training Framework**: Bitter Retrieval
- **Base Model**: {model_path}

## Training Configuration

"""
    
    if experiment_info:
        model_card += f"""
- **Experiment**: {experiment_info.get('experiment', 'Custom')}
- **Loss Components**: {experiment_info.get('loss_components', 'Unknown')}
- **Training Method**: {experiment_info.get('training_method', 'Unknown')}
"""
    
    model_card += """
## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{}").cuda()
tokenizer = AutoTokenizer.from_pretrained("{}")

# Encode query
query = "What is machine learning?"
inputs = tokenizer(f"query: {{query}}", return_tensors="pt").to("cuda")
query_emb = model(**inputs).last_hidden_state.mean(dim=1)

# Encode passage  
passage = "Machine learning is a subset of artificial intelligence..."
inputs = tokenizer(f"passage: {{passage}}", return_tensors="pt").to("cuda")
passage_emb = model(**inputs).last_hidden_state.mean(dim=1)

# Compute similarity
similarity = torch.cosine_similarity(query_emb, passage_emb)
```

## Citation

If you use this model, please cite the Bitter Retrieval framework.
""".format(repo_name, repo_name)
    
    return model_card


def main():
    parser = argparse.ArgumentParser(description="Push models and datasets to HuggingFace Hub")
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--dataset", type=str, help="Path to dataset file")
    parser.add_argument("--repo", type=str, help="HuggingFace repository name (username/repo-name)")
    parser.add_argument("--type", choices=["model", "dataset"], default="model", help="Type of content to push")
    parser.add_argument("--split", default="train", help="Dataset split name")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--message", type=str, help="Commit message")
    parser.add_argument("--list-local", action="store_true", help="List available local models")
    parser.add_argument("--create-card", action="store_true", help="Create model card")
    
    args = parser.parse_args()
    
    print("🤗 HuggingFace Operations for Bitter Retrieval")
    print("=" * 50)
    
    if args.list_local:
        list_local_models()
        return
    
    if not args.repo:
        print("❌ Repository name required (--repo username/repo-name)")
        return
    
    success = False
    
    if args.type == "model" or args.model:
        if not args.model:
            print("❌ Model path required (--model path/to/model)")
            return
        
        success = push_model_to_hf(
            args.model, 
            args.repo, 
            private=args.private,
            commit_message=args.message
        )
        
        if success and args.create_card:
            print("📝 Creating model card...")
            card_content = create_model_card(args.repo, args.model)
            print("Model card preview:")
            print("-" * 30)
            print(card_content[:500] + "..." if len(card_content) > 500 else card_content)
    
    elif args.type == "dataset" or args.dataset:
        if not args.dataset:
            print("❌ Dataset path required (--dataset path/to/dataset)")
            return
        
        success = push_dataset_to_hf(
            args.dataset,
            args.repo,
            split_name=args.split,
            private=args.private,
            commit_message=args.message
        )
    
    if success:
        print(f"\n🎉 Successfully uploaded to HuggingFace!")
        print(f"🔗 View at: https://huggingface.co/{args.repo}")
    else:
        print("\n❌ Upload failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 