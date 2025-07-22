#!/usr/bin/env python3
"""
Helper script to set up API keys for bitter-retrieval.
"""

import os
from pathlib import Path


def setup_env_file():
    """Interactive setup of .env file."""
    env_path = Path(".env")
    example_path = Path(".env.example")
    
    print("🔑 Bitter Retrieval API Keys Setup")
    print("=" * 40)
    
    # Check if .env already exists
    if env_path.exists():
        print("✅ .env file already exists")
        response = input("Do you want to update it? [y/N]: ").lower().strip()
        if response != 'y':
            print("Keeping existing .env file")
            return
    
    # Copy from example if it doesn't exist
    if not env_path.exists() and example_path.exists():
        import shutil
        shutil.copy(example_path, env_path)
        print("📄 Created .env from template")
    
    print("\n📝 We'll help you set up your API keys...")
    
    # Hugging Face Token
    print("\n1️⃣  Hugging Face Token (REQUIRED)")
    print("   This is needed to access gated models like Llama")
    print("   👉 Get your token at: https://huggingface.co/settings/tokens")
    print("   ℹ️  Create a token with 'Read' access")
    
    hf_token = input("\n   Enter your Hugging Face token (or press Enter to skip): ").strip()
    
    # Weights & Biases API Key
    print("\n2️⃣  Weights & Biases API Key (OPTIONAL)")
    print("   This enables experiment tracking and monitoring")
    print("   👉 Get your key at: https://wandb.ai/authorize")
    
    wandb_key = input("\n   Enter your W&B API key (or press Enter to skip): ").strip()
    
    # Optional: W&B settings
    wandb_entity = ""
    if wandb_key:
        print("\n3️⃣  Weights & Biases Settings")
        wandb_entity = input("   Enter your W&B username/entity (optional): ").strip()
    
    # Update .env file
    env_content = []
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            env_content = f.readlines()
    
    # Create new content
    new_content = []
    updated_keys = set()
    
    for line in env_content:
        if line.startswith('HUGGINGFACE_TOKEN=') and hf_token:
            new_content.append(f"HUGGINGFACE_TOKEN={hf_token}\n")
            updated_keys.add('HUGGINGFACE_TOKEN')
        elif line.startswith('WANDB_API_KEY=') and wandb_key:
            new_content.append(f"WANDB_API_KEY={wandb_key}\n")
            updated_keys.add('WANDB_API_KEY')
        elif line.startswith('WANDB_ENTITY=') and wandb_entity:
            new_content.append(f"WANDB_ENTITY={wandb_entity}\n")
            updated_keys.add('WANDB_ENTITY')
        else:
            new_content.append(line)
    
    # Add new keys if they weren't in the file
    if hf_token and 'HUGGINGFACE_TOKEN' not in updated_keys:
        new_content.append(f"HUGGINGFACE_TOKEN={hf_token}\n")
    
    if wandb_key and 'WANDB_API_KEY' not in updated_keys:
        new_content.append(f"WANDB_API_KEY={wandb_key}\n")
    
    if wandb_entity and 'WANDB_ENTITY' not in updated_keys:
        new_content.append(f"WANDB_ENTITY={wandb_entity}\n")
    
    # Write the file
    with open(env_path, 'w') as f:
        f.writelines(new_content)
    
    print("\n" + "=" * 40)
    print("✅ API keys configured successfully!")
    
    if hf_token:
        print("✅ Hugging Face: Configured")
    else:
        print("❌ Hugging Face: Not configured (required for training)")
    
    if wandb_key:
        print("✅ Weights & Biases: Configured")
    else:
        print("⚠️  Weights & Biases: Not configured (optional)")
    
    print("\n💡 Next steps:")
    print("1. Test your setup: poetry run python test_setup.py")
    print("2. Run training: ./run_training.sh <google_drive_file_id>")
    
    if not hf_token:
        print("\n⚠️  Warning: You still need to set HUGGINGFACE_TOKEN to train models!")


if __name__ == "__main__":
    setup_env_file() 