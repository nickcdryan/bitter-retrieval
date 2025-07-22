"""
Authentication and API key management for bitter-retrieval.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_env_file() -> None:
    """Load environment variables from .env file if it exists."""
    try:
        from dotenv import load_dotenv
        
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Loaded environment variables from .env file")
        else:
            logger.info("No .env file found, using system environment variables")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")


def setup_huggingface_auth() -> bool:
    """Setup Hugging Face authentication."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not token or token == "your_huggingface_token_here":
        logger.error("HUGGINGFACE_TOKEN not set or using placeholder value")
        logger.error("Please set your Hugging Face token in .env file")
        logger.error("Get your token from: https://huggingface.co/settings/tokens")
        return False
    
    try:
        from huggingface_hub import login
        login(token=token)
        logger.info("Successfully authenticated with Hugging Face")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        return False


def setup_wandb_auth() -> bool:
    """Setup Weights & Biases authentication."""
    api_key = os.getenv("WANDB_API_KEY")
    
    if not api_key or api_key == "your_wandb_api_key_here":
        logger.warning("WANDB_API_KEY not set, W&B logging will be disabled")
        return False
    
    try:
        import wandb
        wandb.login(key=api_key)
        logger.info("Successfully authenticated with Weights & Biases")
        return True
    except Exception as e:
        logger.warning(f"Failed to authenticate with W&B: {e}")
        return False


def setup_authentication() -> dict:
    """
    Setup all authentication and return status.
    
    Returns:
        dict: Authentication status for each service
    """
    # Load environment variables
    load_env_file()
    
    auth_status = {
        "huggingface": setup_huggingface_auth(),
        "wandb": setup_wandb_auth()
    }
    
    return auth_status


def check_env_setup() -> None:
    """Check if .env file is properly configured."""
    env_path = Path(".env")
    example_path = Path(".env.example")
    
    if not env_path.exists():
        print("\n🔑 API Keys Setup Required")
        print("=" * 40)
        print("❌ No .env file found!")
        print("\n📝 To set up your API keys:")
        print("1. Copy the example file:")
        print("   cp .env.example .env")
        print("\n2. Edit .env and add your API keys:")
        print("   - Hugging Face token: https://huggingface.co/settings/tokens")
        print("   - Weights & Biases key: https://wandb.ai/authorize")
        print("\n3. Run the training script again")
        return
    
    # Check if tokens are still placeholder values
    load_env_file()
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
    wandb_key = os.getenv("WANDB_API_KEY", "")
    
    issues = []
    if not hf_token or hf_token == "your_huggingface_token_here":
        issues.append("❌ HUGGINGFACE_TOKEN not set")
    
    if not wandb_key or wandb_key == "your_wandb_api_key_here":
        issues.append("⚠️  WANDB_API_KEY not set (optional)")
    
    if issues:
        print("\n🔑 API Keys Configuration Issues")
        print("=" * 40)
        for issue in issues:
            print(issue)
        print("\n📝 Please edit your .env file and set the required tokens")
    else:
        print("✅ API keys configured properly") 