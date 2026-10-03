#!/usr/bin/env python3
"""Download the pinned official model snapshot; do not persist credentials."""
import argparse
from pathlib import Path

MODEL = "Qwen/Qwen3-VL-32B-Instruct"
REVISION = "0cfaf48183f594c314753d30a4c4974bc75f3ccb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    location = snapshot_download(repo_id=MODEL, revision=REVISION,
                                 local_dir=str(args.local_dir.resolve()))
    print(f"Downloaded {MODEL}@{REVISION} to {location}")


if __name__ == "__main__":
    main()

