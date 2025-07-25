#!/usr/bin/env python3

# python push_to_huggingface.py

"""
Simple utility to push trained models to HuggingFace Hub
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("python-dotenv not installed - using system environment variables only")

# CONFIGURE THESE VARIABLES
MODEL_PATH = "models/kl-margin:3-gradclip-temp.01-Nomic-fulltraining-epoch:2-batch:16_modular_kl_0.5_margin_0.5/"  # Change this to your model path
REPO_NAME = "bitter-retrieval-kl-margin-nomic"  # Change this to your desired repo name
PRIVATE = False  # Set to True if you want a private repo

def push_model():
    """Push model to HuggingFace Hub"""
    
    # Login
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("No HF_TOKEN found. Please set environment variable or login with: huggingface-cli login")
        return
    
    # Get username
    api = HfApi()
    try:
        username = api.whoami()["name"]
        full_repo_name = f"{username}/{REPO_NAME}"
    except:
        print("Could not get username. Please check your HF token.")
        return
    
    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"Model path {MODEL_PATH} does not exist")
        return
    
    # Push to hub
    try:
        api.create_repo(repo_id=full_repo_name, private=PRIVATE, exist_ok=True)
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=full_repo_name,
            commit_message="Upload model"
        )
        print(f"✅ Pushed to https://huggingface.co/{full_repo_name}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    push_model() 