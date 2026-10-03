#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Linux ]] || { echo 'Inference requires Linux x86_64 and NVIDIA CUDA GPUs.' >&2; exit 1; }
[[ ! -e .venv ]] || { echo '.venv already exists; use a clean checkout to avoid carrying old patches.' >&2; exit 1; }
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
# Historical pip-freeze, including the cuDNN override. Read environment/README.md.
.venv/bin/python -m pip install --no-deps -r environment/requirements.lock.txt
.venv/bin/python -c 'import torch,sglang; print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available())'

