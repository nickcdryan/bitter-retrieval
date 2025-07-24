#!/usr/bin/env python3

"""
Script to push trained retrieval models to HuggingFace Hub
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login
from transformers import AutoModel, AutoTokenizer
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("python-dotenv not installed - using system environment variables only")


def create_model_card(model_name, training_method, base_model="google-bert/bert-base-uncased"):
    """Create a comprehensive model card for the trained model"""
    
    model_card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- retrieval
- information-retrieval
- sentence-transformers
- bert
- msmarco
- squad
pipeline_tag: feature-extraction
---

# {model_name}

This is a retrieval model fine-tuned using **{training_method}** on MS MARCO dataset with additional validation on SQuAD.

## Model Details

- **Base Model**: {base_model}
- **Training Method**: {training_method}
- **Training Data**: MS MARCO soft-labeled dataset
- **Validation Data**: SQuAD v2 + MS MARCO
- **Framework**: PyTorch + Transformers

## Training Details

This model was trained using the bitter-retrieval framework with:

- **Training Method**: `{training_method}`
- **Encoder**: BERT-base-uncased
- **Max Sequence Length**: 512 tokens
- **Batch Size**: 32
- **Epochs**: 2
- **Learning Rate**: 2e-5
- **Temperature**: 0.02

## Usage

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

# Load model and tokenizer
model = AutoModel.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

def encode_text(text, prefix=""):
    '''Encode text with optional prefix'''
    full_text = f"{{prefix}}{{text}}" if prefix else text
    inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        count_tokens = attention_mask.sum(dim=1, keepdim=True)
        embeddings = sum_embeddings / count_tokens
        # L2 normalize
        embeddings = F.normalize(embeddings, dim=-1)
    
    return embeddings

# Example usage
query = "What is machine learning?"
passage = "Machine learning is a subset of artificial intelligence..."

# Encode with prefixes (recommended)
query_emb = encode_text(query, "query: ")
passage_emb = encode_text(passage, "passage: ")

# Compute similarity
similarity = torch.cosine_similarity(query_emb, passage_emb)
print(f"Similarity: {{similarity.item():.4f}}")
```

## Evaluation Metrics

The model was evaluated on both SQuAD and MS MARCO datasets with the following metrics:
- **Retrieval Accuracy**: How often the correct passage is retrieved
- **F1 Score**: Token-level F1 between generated and reference answers
- **Exact Match**: Exact match between generated and reference answers
- **LLM Judge**: Semantic similarity judged by Gemini-2.0-flash

## Training Framework

This model was trained using the [bitter-retrieval](https://github.com/yourusername/bitter-retrieval) framework, which implements various contrastive learning methods for retrieval tasks.

## Citation

If you use this model, please cite:

```bibtex
@misc{{bitter-retrieval-{training_method.lower().replace('_', '-')},
  title={{Bitter Retrieval: {training_method} Fine-tuned BERT for Information Retrieval}},
  author={{Your Name}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/{model_name}}}}}
}}
```
"""
    
    return model_card


def push_model_to_hub(local_path, repo_name, training_method, private=False):
    """Push a model directory to HuggingFace Hub"""
    
    print(f"📤 Pushing {local_path} to {repo_name}")
    
    # Check if model files exist
    model_path = Path(local_path)
    if not model_path.exists():
        print(f"❌ Model path {local_path} does not exist")
        return False
    
    config_file = model_path / "config.json"
    model_file = model_path / "model.safetensors"
    
    if not config_file.exists() or not model_file.exists():
        print(f"❌ Required files missing in {local_path}")
        return False
    
    try:
        # Load the model to verify it works
        print("🔍 Verifying model can be loaded...")
        model = AutoModel.from_pretrained(local_path)
        print("✅ Model loaded successfully")
        
        # Create model card
        model_card_content = create_model_card(repo_name, training_method)
        
        # Save model card
        readme_path = model_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(model_card_content)
        print("✅ Model card created")
        
        # Push to hub
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_name, private=private, exist_ok=True)
            print(f"✅ Repository {repo_name} ready")
        except Exception as e:
            print(f"⚠️  Repository creation warning: {e}")
        
        # Upload all files
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            commit_message=f"Upload {training_method} retrieval model"
        )
        
        print(f"🎉 Successfully pushed to https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing model: {e}")
        return False


def main():
    """Main function to push models to HuggingFace"""
    
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        print("🔑 Logging into HuggingFace...")
        login(token=hf_token)
    else:
        print("⚠️  No HF_TOKEN found. Please login manually with `huggingface-cli login`")
        return
    
    # Get username for repository naming
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"👤 Logged in as: {username}")
    except:
        username = input("Enter your HuggingFace username: ")
    
    # Define models to push
    models_dir = Path("models")
    models_to_push = [
        {
            "local_path": "models/converted_infonce-BERT-fulltraining-epoch:2-batch:32",
            "repo_name": f"{username}/bitter-retrieval-converted-infonce-bert",
            "training_method": "Converted InfoNCE"
        },
        {
            "local_path": "models/standard_infonce-BERT-fulltraining-epoch:2-batch:32", 
            "repo_name": f"{username}/bitter-retrieval-standard-infonce-bert",
            "training_method": "Standard InfoNCE"
        }
    ]
    
    # Ask user which models to push
    print("\\n📋 Available models to push:")
    for i, model_info in enumerate(models_to_push):
        print(f"{i+1}. {model_info['local_path']} -> {model_info['repo_name']}")
    
    choice = input("\\nEnter model numbers to push (e.g., '1,2' for both, or '1' for first): ").strip()
    
    if choice.lower() in ['all', 'both', '1,2', '2,1']:
        selected_models = models_to_push
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_models = [models_to_push[i] for i in indices if 0 <= i < len(models_to_push)]
        except:
            print("❌ Invalid selection. Pushing all models.")
            selected_models = models_to_push
    
    # Ask if models should be private
    make_private = input("\\nMake repositories private? (y/N): ").lower().startswith('y')
    
    # Push selected models
    success_count = 0
    for model_info in selected_models:
        print(f"\\n{'='*60}")
        success = push_model_to_hub(
            model_info["local_path"],
            model_info["repo_name"], 
            model_info["training_method"],
            private=make_private
        )
        if success:
            success_count += 1
    
    print(f"\\n🏁 Complete! Successfully pushed {success_count}/{len(selected_models)} models")
    
    if success_count > 0:
        print("\\n🔗 Your models are now available at:")
        for model_info in selected_models:
            print(f"   https://huggingface.co/{model_info['repo_name']}")


if __name__ == "__main__":
    main() 