"""
Automated 1-Click Hugging Face Space Uploader
Uses huggingface_hub to automatically handle Git LFS for large videos and model weights.
"""

import os
import sys
from huggingface_hub import HfApi

def main():
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
    else:
        token = input("Enter your Hugging Face Token (hf_...): ").strip()

    if not token:
        print("❌ Error: Token is required.")
        sys.exit(1)

    print("🚀 Uploading to Hugging Face Space: Neolithihuman/Calance...")
    print("📦 Handling large files (.mp4 videos, .pt model weights) with automatic Git LFS...")
    
    api = HfApi(token=token)
    
    api.upload_folder(
        folder_path=".",
        repo_id="Neolithihuman/Calance",
        repo_type="space",
        ignore_patterns=[
            "venv/**",
            "__pycache__/**",
            ".git/**",
            "*.pyc",
            ".DS_Store",
            "uploads/*",
            "!uploads/.gitkeep",
            "yolo11n.pt",
            "screenshot*.png"
        ]
    )
    print("\n✅ Successfully uploaded all files and models to Hugging Face!")
    print("🌐 View your running app at: https://huggingface.co/spaces/Neolithihuman/Calance\n")

if __name__ == "__main__":
    main()
