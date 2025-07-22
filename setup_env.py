#!/usr/bin/env python3
"""
Helper script to set up API keys and Git configuration for bitter-retrieval.
"""

import os
import subprocess
from pathlib import Path


def setup_git_config():
    """Interactive setup of Git configuration."""
    print("\n🔧 Git Configuration Setup")
    print("=" * 40)
    
    # Check if git is already configured
    try:
        result = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True)
        current_name = result.stdout.strip() if result.returncode == 0 else None
        
        result = subprocess.run(['git', 'config', 'user.email'], capture_output=True, text=True)
        current_email = result.stdout.strip() if result.returncode == 0 else None
        
        if current_name and current_email:
            print(f"✅ Git already configured:")
            print(f"   Name: {current_name}")
            print(f"   Email: {current_email}")
            response = input("\nDo you want to update Git config? [y/N]: ").lower().strip()
            if response != 'y':
                print("Keeping existing Git configuration")
                return
    except FileNotFoundError:
        print("❌ Git not found. Please install Git first.")
        return
    
    print("\n📝 Setting up Git configuration...")
    print("This is needed to make commits and push changes.")
    
    # Get user input
    if current_name:
        name = input(f"\nEnter your name [{current_name}]: ").strip()
        if not name:
            name = current_name
    else:
        name = input("\nEnter your name: ").strip()
    
    if current_email:
        email = input(f"Enter your email [{current_email}]: ").strip()
        if not email:
            email = current_email
    else:
        email = input("Enter your email: ").strip()
    
    if not name or not email:
        print("❌ Name and email are required for Git configuration")
        return
    
    try:
        # Set global Git configuration
        subprocess.run(['git', 'config', '--global', 'user.name', name], check=True)
        subprocess.run(['git', 'config', '--global', 'user.email', email], check=True)
        
        print(f"\n✅ Git configured successfully!")
        print(f"   Name: {name}")
        print(f"   Email: {email}")
        
        # Show recommended workflow
        print(f"\n💡 Recommended workflow for pushing changes:")
        print("   git checkout -b feature/my-changes")
        print("   git add .")
        print("   git commit -m 'Description of changes'")
        print("   git push origin feature/my-changes")
        print("   # Then create a Pull Request on GitHub")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to configure Git: {e}")


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
        wandb_entity = input("   Enter your W&B username (optional): ").strip()
    
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


def main():
    """Main setup function that runs both API key and Git setup."""
    print("🚀 Bitter Retrieval Setup")
    print("=" * 50)
    print("This script will help you set up:")
    print("• API keys (.env file)")
    print("• Git configuration")
    
    # Setup API keys
    setup_env_file()
    
    # Setup Git
    setup_git_config()
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\n💡 Next steps:")
    print("1. Test your setup: poetry run python test_setup.py")
    print("2. Run training: ./run_training.sh <google_drive_file_id>")
    print("3. Make changes and push with: git checkout -b your-branch-name")


if __name__ == "__main__":
    main() 