"""
Model Weights Downloader
Downloads pretrained head-detection weights into the models directory.
"""

import os
import urllib.request
import logging

from config import MODEL_REGISTRY, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("DownloadModels")


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        logger.info(f"File {dest_path} already exists ({os.path.getsize(dest_path)/(1024*1024):.2f} MB). Skipping.")
        return

    logger.info(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    )
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        total = resp.getheader("content-length")
        total = int(total) if total else None
        downloaded = 0
        chunk_size = 1024 * 512
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            downloaded += len(chunk)
            f.write(chunk)
            if total:
                pct = (downloaded / total) * 100
                print(f"\rDownloading {os.path.basename(dest_path)}: {pct:.1f}% ({downloaded/(1024*1024):.1f}/{total/(1024*1024):.1f} MB)", end="")
        print()
    logger.info(f"Saved {dest_path}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for model_key, info in MODEL_REGISTRY.items():
        url = info.get("url")
        filename = info.get("filename")
        if url and filename:
            dest = os.path.join(MODELS_DIR, filename)
            try:
                download_file(url, dest)
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")


if __name__ == "__main__":
    main()
