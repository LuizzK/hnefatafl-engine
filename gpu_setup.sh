#!/bin/bash
# GPU Instance Setup Script for Copenhagen Hnefatafl Training
# Run this on your GPU instance after uploading the code

set -e  # Exit on any error

echo "========================================="
echo "Copenhagen Hnefatafl GPU Setup"
echo "========================================="

# 1. Extract code
echo ""
echo "1. Extracting code..."
tar -xzf hnefatafl_upload.tar.gz
cd copenhagen-hnefatafl 2>/dev/null || true  # cd if in subdirectory

# 2. Check Python version
echo ""
echo "2. Checking Python..."
python3 --version

# 3. Check CUDA availability
echo ""
echo "3. Checking CUDA..."
nvidia-smi

# 4. Install dependencies
echo ""
echo "4. Installing dependencies..."
pip3 install --no-cache-dir torch numpy pygame tensorboard tqdm pytest

# 5. Verify PyTorch can see GPU
echo ""
echo "5. Verifying PyTorch + GPU..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✓ GPU ready!')
else:
    print('✗ WARNING: CUDA not available!')
    exit(1)
"

# 6. Quick test run (1 iteration)
echo ""
echo "6. Running quick test (1 iteration)..."
python3 train_main.py --config gpu_test --iterations 1 --yes

echo ""
echo "========================================="
echo "Setup complete! GPU is ready!"
echo "========================================="
echo ""
echo "To start 5-iteration training:"
echo "  python3 train_main.py --config gpu_test --iterations 5 --yes"
echo ""
echo "Estimated cost: ~$0.50-1.00"
echo "Estimated time: ~30-60 minutes"
echo ""
