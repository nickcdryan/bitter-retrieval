#!/usr/bin/env python3

"""
Simple utility to push trained models to HuggingFace Hub
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login

# CONFIGURE THESE VARIABLES
MODEL_PATH = "models/converted_infonce-BERT-fulltraining-epoch:2-batch:32"  # Change this to your model path
REPO_NAME = "bitter-retrieval-converted-infonce-bert"  # Change this to your desired repo name
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