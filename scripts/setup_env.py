#!/usr/bin/env python3
"""
Environment setup script for bitter retrieval training

This script sets up a new training environment with all required dependencies
and configurations for bitter retrieval model training.

Usage:
    python scripts/setup_env.py
    python scripts/setup_env.py --install-flash-attention
    python scripts/setup_env.py --gpu-check-only
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(cmd, description="", check=True):
    """Run a shell command with error handling"""
    print(f"🔧 {description}")
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        if isinstance(cmd, list):
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr and not check:  # Only show stderr if we're not checking
            print(f"Warning: {result.stderr}")
        
        print(f"✅ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return e


def check_gpu():
    """Check GPU availability and specs"""
    print("🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ {gpu_count} GPU(s) available")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ No GPU available - training will be very slow")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU")
        return False
    
    return True


def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "models",
        "data", 
        "logs",
        "configs"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  Created: {dir_name}/")
    
    print("✅ Directories created")


def create_env_template():
    """Create .env template file"""
    print("🔐 Creating .env template...")
    
    env_template = """# Environment variables for bitter retrieval training
# Copy this to .env and fill in your actual values

# Hugging Face token for model access
HUGGINGFACE_TOKEN=your_huggingface_token_here
HF_TOKEN=your_huggingface_token_here

# Weights & Biases API key for logging
WANDB_API_KEY=your_wandb_api_key_here

# Gemini API key for LLM judge evaluation
GEMINI_API_KEY=your_gemini_api_key_here
"""
    
    env_path = Path(".env.example")
    with open(env_path, "w") as f:
        f.write(env_template)
    
    print(f"✅ Created {env_path}")
    print("📝 Edit .env.example and rename to .env with your actual API keys")


def install_flash_attention():
    """Install Flash Attention 2 for faster training"""
    print("⚡ Installing Flash Attention 2...")
    
    # Check if already installed
    try:
        import flash_attn
        print("✅ Flash Attention 2 already installed")
        return True
    except ImportError:
        pass
    
    cmd = [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"]
    result = run_command(cmd, "Installing Flash Attention 2", check=False)
    
    if isinstance(result, subprocess.CalledProcessError):
        print("⚠️  Flash Attention 2 installation failed - this is optional but recommended for speed")
        return False
    
    return True


def check_dependencies():
    """Check if key dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "wandb",
        "tqdm",
        "numpy",
        "pyyaml",
        "google-generativeai"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"📥 Missing packages: {missing}")
        print("Install them with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies installed")
    return True


def create_example_config():
    """Create example configuration files"""
    print("⚙️  Creating example configurations...")
    
    # Simple experiment config
    example_config = {
        "experiment_name": "kl_margin_example",
        "model_config": {
            "encoder_model": "nomic-ai/nomic-embed-text-v1-unsupervised",
            "llm_model": "Qwen/Qwen3-8B-Base"
        },
        "training_config": {
            "training_method": "modular",
            "loss_components": {"kl": 0.5, "margin": 0.5},
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 2
        },
        "data_config": {
            "dataset_name": "nickcdryan/ms_marco_softlabel_Qwen3-8B-Base_bf16",
            "num_data_examples": 1000
        }
    }
    
    config_path = Path("configs/example_experiment.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(example_config, f, indent=2)
    
    print(f"✅ Created {config_path}")


def print_usage_instructions():
    """Print instructions for using the repository"""
    print("\n🚀 Setup completed! Here's how to use bitter retrieval:")
    print("=" * 60)
    
    print("\n📝 1. Configure environment:")
    print("   - Copy .env.example to .env")
    print("   - Add your API keys (HuggingFace, W&B, Gemini)")
    
    print("\n🎯 2. Train a model:")
    print("   python scripts/train.py --experiment kl_margin")
    print("   python scripts/train.py --run-name my-experiment --batch-size 32")
    
    print("\n🏷️  3. Label new data:")
    print("   python scripts/label_data.py --model Qwen/Qwen3-8B-Base --split train --num 1000")
    
    print("\n📊 4. Evaluate a model:")
    print("   python scripts/evaluate.py --model-path models/my-model")
    print("   python scripts/evaluate.py --encoder nomic-ai/nomic-embed-text-v1")
    
    print("\n📚 5. Available experiments:")
    print("   - kl_only: KL divergence loss only")
    print("   - margin_only: Margin loss only") 
    print("   - kl_margin: KL + margin combination")
    print("   - infonce_standard: Standard InfoNCE")
    print("   - infonce_converted: Converted InfoNCE")
    
    print("\n🔧 6. Custom training:")
    print("   - Edit bitter_retrieval/config.py for new experiments")
    print("   - Modify loss components in modular training")
    print("   - Add new loss functions in bitter_retrieval/training/losses.py")


def main():
    parser = argparse.ArgumentParser(description="Setup environment for bitter retrieval training")
    parser.add_argument("--install-flash-attention", action="store_true", help="Install Flash Attention 2")
    parser.add_argument("--gpu-check-only", action="store_true", help="Only check GPU availability")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency checking")
    
    args = parser.parse_args()
    
    print("🎯 Bitter Retrieval Environment Setup")
    print("=" * 40)
    
    # GPU check
    gpu_available = check_gpu()
    
    if args.gpu_check_only:
        return
    
    # Dependencies
    if not args.skip_deps:
        deps_ok = check_dependencies()
        if not deps_ok:
            print("\n❌ Please install missing dependencies first")
            return
    
    # Setup
    setup_directories()
    create_env_template()
    create_example_config()
    
    # Optional Flash Attention
    if args.install_flash_attention or (gpu_available and input("\n⚡ Install Flash Attention 2 for faster training? (y/n): ").lower() == 'y'):
        install_flash_attention()
    
    # Usage instructions
    print_usage_instructions()
    
    print("\n✅ Environment setup completed!")


if __name__ == "__main__":
    main() 