#!/bin/bash
# Creates the incontext-bayesians conda environment.
# Run once from the repo root:
#   bash setup_env.sh
#
# After it finishes, activate with:
#   conda activate incontext-bayesians

set -e

CONDA=/home/katie/miniconda3/bin/conda
ENV_NAME=incontext-bayesians

echo "==> Creating conda environment '$ENV_NAME' (Python 3.11)..."
$CONDA create -n $ENV_NAME python=3.11 -y

echo "==> Installing PyTorch 2.4.1 with CUDA 12.1..."
$CONDA run -n $ENV_NAME pip install torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing TransformerLens and dependencies..."
$CONDA run -n $ENV_NAME pip install \
    transformer_lens==2.16.1 \
    transformers>=4.43.0 \
    accelerate>=0.33.0 \
    einops \
    jaxtyping \
    numpy \
    matplotlib \
    scienceplots \
    plotly \
    tqdm \
    requests \
    huggingface_hub

echo "==> Verifying PyTorch + CUDA..."
$CONDA run -n $ENV_NAME python -c "
import torch
print(f'  torch version : {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU           : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "==> Done! Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "==> Then log in to HuggingFace (required for Llama-3.1-8B):"
echo "    huggingface-cli login"
