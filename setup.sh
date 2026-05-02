#!/bin/bash
# Setup script for contrastive-dcot on Lightning AI (CUDA 12.4 / A100).
# Run once on a fresh environment: bash setup.sh
#
# Order matters:
#   1. Install all packages including vllm (vllm will pull torch 2.4, that's ok for now)
#   2. Force-reinstall the correct torch cu124 stack at the end
#      (this overwrites whatever vllm dragged in)

set -e

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Pinning torch to 2.5.1+cu124 (overrides vllm's torch preference)..."
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Verifying..."
python -c "
import torch, transformers, accelerate, bitsandbytes, vllm
print('torch:         ', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('transformers:  ', transformers.__version__)
print('accelerate:    ', accelerate.__version__)
print('bitsandbytes:  ', bitsandbytes.__version__)
print('vllm:          ', vllm.__version__)
"

echo "==> Done."
