#!/usr/bin/env python3
"""
Download soft-labeled data from Google Drive for bitter-retrieval training.
"""

import os
import sys
from pathlib import Path

def download_from_google_drive(file_id: str, output_path: str):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download file
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive to {output_path}...")
    gdown.download(url, output_path, quiet=False)
    
    # Verify download
    if not Path(output_path).exists():
        raise FileNotFoundError(f"Download failed: {output_path}")
    
    print(f"✅ Successfully downloaded to {output_path}")
    return output_path


def main():
    """Main function to download data."""
    if len(sys.argv) != 2:
        print("Usage: python download_data.py <google_drive_file_id>")
        print("Example: python download_data.py 1ABCdef123xyz...")
        print("\nTo get the file ID from a Google Drive link:")
        print("https://drive.google.com/file/d/1ABCdef123xyz.../view -> 1ABCdef123xyz...")
        sys.exit(1)
    
    file_id = sys.argv[1]
    output_path = "data/msmarco/soft_labels_msmarco_3B_nopad_5000.json"
    
    try:
        download_from_google_drive(file_id, output_path)
        print(f"\n🎉 Data ready! You can now run training with:")
        print("python -m src.bitter_retrieval.train --method standard_infonce")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 