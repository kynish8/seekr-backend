#!/bin/bash
source ~/seekr-backend/venv/bin/activate
if nvidia-smi &>/dev/null; then
    echo "[setup] GPU detected — installing CUDA PyTorch..."
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "[setup] No GPU — installing CPU PyTorch..."
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
echo "[setup] Done. Run: sudo systemctl restart seekr"
