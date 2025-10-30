# GPU Test Training Instructions

## What This Is

A small GPU training run with:
- **5 iterations**
- **5 games per iteration** (25 games total)
- **1000 move limit** (attackers win on timeout)
- **Standard model** (128 channels, 10 blocks)
- **800 MCTS simulations per move**

**Time:** ~30-60 minutes
**Cost:** ~$0.20-0.50
**Purpose:** Verify GPU works before committing to full training

## Setup on GPU

### 1. Rent GPU (RunPod or Vast.ai)

**RunPod (Recommended):**
1. Go to https://runpod.io
2. Sign up and add $5-10 credit
3. Select "Community Cloud"
4. Deploy RTX 4090 with PyTorch template
5. Connect via SSH

**Vast.ai (Budget):**
1. Go to https://vast.ai
2. Search for RTX 4090 or RTX 3090
3. Select instance with good reliability (95%+)
4. Launch with PyTorch template
5. Connect via SSH

### 2. Clone and Setup

```bash
# Install git if needed
apt-get update && apt-get install -y git

# Clone repo
cd ~
git clone https://github.com/YourUsername/copenhagen-hnefatafl.git
cd copenhagen-hnefatafl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch for CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
# Should print: GPU: True
```

### 3. Run GPU Test Training

```bash
# Start training with 5 iterations
nohup python train_main.py --config gpu_test --iterations 5 --yes > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### 4. What You'll See

```
======================================================================
COPENHAGEN HNEFATAFL
                  AlphaZero Training
======================================================================

Loading 'gpu_test' configuration...

Initializing trainer...
Device: cuda
Model parameters: 19,036,705

======================================================================
TRAINING PLAN
======================================================================

Iterations: 5
Device: cuda

Estimated totals:
  Games: ~25
  Positions: ~5,000-10,000

Key parameters:
  MCTS simulations: 800
  Games per iteration: 5
  Max moves per game: 1000
  Attacker timeout win: True
  Batch size: 256
  Model size: 128 channels, 10 blocks

======================================================================

ITERATION 1
======================================================================

[1/5] Generating self-play games...

  Starting game 1/5...
    Move 5 (Attacker's turn)
    Move 10 (Defender's turn)
    ... continues ...
```

### 5. Expected Results

**Each iteration should:**
- Generate 5 games (~5-10 minutes)
- Train neural network (~1-2 minutes)
- Evaluate model (~2-3 minutes)
- Save checkpoint (~1 second)

**Total time:** ~30-60 minutes

**Success indicators:**
- ✅ All 5 iterations complete
- ✅ Loss decreases over iterations
- ✅ No NaN or infinity values
- ✅ Checkpoints saved successfully
- ✅ GPU utilization 80-100%

### 6. Download Checkpoints

```bash
# On GPU machine
cd ~/copenhagen-hnefatafl
tar -czf checkpoints.tar.gz checkpoints_gpu_test/

# From your local machine
scp -P XXXXX root@GPU_IP:~/copenhagen-hnefatafl/checkpoints.tar.gz .

# Extract locally
tar -xzf checkpoints.tar.gz
```

### 7. Test the Model

```bash
# On your local machine
./venv/bin/python play.py --game-mode cpu --checkpoint checkpoints_gpu_test/best_model.pt
```

## What This Proves

If this succeeds, you know:
- ✅ GPU access works
- ✅ CUDA is properly configured
- ✅ Training pipeline functions correctly on GPU
- ✅ Checkpoints save and load correctly
- ✅ 1000-move games work (attackers win on timeout)
- ✅ You can safely run longer training

## Next Steps After Success

### Option A: Continue Training
```bash
# Add 50 more iterations to same model
python train_main.py --resume checkpoints_gpu_test/checkpoint_iter_5.pt --iterations 55 --yes
```

### Option B: Start Full Training
```bash
# Start fresh with standard config
python train_main.py --config standard --iterations 500 --yes
# Cost: ~$15-20 for competent play
```

### Option C: Test Model First
Download checkpoints and play against your AI locally to see if it's improving.

## Troubleshooting

### GPU not found
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch for correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of memory
The config is sized for RTX 4090 (24GB VRAM). If using smaller GPU:
- Reduce batch_size in config.py GPUTestConfig (try 128 or 64)
- Or rent a larger GPU

### Training crashes
- Check `training.log` for error message
- Resume from last checkpoint if it saved any
- Report error for debugging

## Cost Breakdown

**Estimated costs:**
- Vast.ai RTX 3090 @ $0.15/hr × 1hr = **$0.15**
- Vast.ai RTX 4090 @ $0.25/hr × 1hr = **$0.25**
- RunPod RTX 4090 @ $0.34/hr × 1hr = **$0.34**

**Very cheap to verify everything works!**

## Summary

This GPU test is designed to:
1. Verify your GPU setup works (30-60 min, $0.20-0.50)
2. Test longer games with realistic rules (1000 moves, attacker timeout)
3. Validate the full training pipeline on real GPU hardware
4. Give you confidence before spending $15-50 on full training

**If this succeeds → you're safe to run full training!**
