# RunPod 5-Iteration Training Guide

Quick guide to run 5-iteration GPU training on RunPod for ~$0.50-1.00.

## Step 1: Launch RunPod Instance

1. Go to https://www.runpod.io/
2. Click "Deploy" → "GPU Pods"
3. **Choose GPU**:
   - **RTX 3060 12GB** (~$0.34/hr) - RECOMMENDED
   - RTX 3070/3080 if 3060 not available
   - Avoid RTX 5090 (needs PyTorch 2.7+)

4. **Template**: PyTorch (or any with Python 3.10+)
5. **Storage**: 20GB is plenty
6. Click "Deploy"

**Cost**: ~$0.50-1.00 for 5 iterations (1-2 hours)

## Step 2: Upload Files

Once your pod is running:

1. Click "Connect" → "Start Web Terminal" (or SSH)
2. In the terminal, create a working directory:
   ```bash
   mkdir -p ~/hnefatafl
   cd ~/hnefatafl
   ```

3. Upload the archive (choose one method):

   **Method A: Use RunPod's web file upload**
   - Click "Connect" → "HTTP Service" → File browser
   - Upload `hnefatafl_upload.tar.gz`
   - Move to ~/hnefatafl/

   **Method B: Use SCP from your local machine**
   ```bash
   # Get SSH command from RunPod (under "Connect" → "SSH")
   # It will look like: ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_ed25519

   # From your local machine:
   scp -P XXXXX -i ~/.ssh/id_ed25519 \
     hnefatafl_upload.tar.gz \
     root@X.X.X.X:~/hnefatafl/
   ```

   **Method C: Clone from GitHub (if you pushed)**
   ```bash
   git clone https://github.com/YOUR_USERNAME/copenhagen-hnefatafl.git
   cd copenhagen-hnefatafl
   ```

## Step 3: Setup and Verify

In the RunPod terminal:

```bash
cd ~/hnefatafl

# Extract files
tar -xzf hnefatafl_upload.tar.gz

# Check CUDA
nvidia-smi

# Install dependencies
pip3 install torch numpy pygame tensorboard tqdm pytest

# Verify GPU works with PyTorch
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

You should see:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

## Step 4: Run Quick Test (1 iteration)

First, verify everything works with 1 iteration (~10 min):

```bash
python3 train_main.py --config gpu_test --iterations 1 --yes
```

Watch for:
- ✓ "Self-play: Game 1/5" (should be fast on GPU)
- ✓ "Training network..." (should complete in seconds)
- ✓ "Checkpoint saved"

If this works, you're ready!

## Step 5: Run 5-Iteration Training

Start the full 5-iteration run:

```bash
# Option 1: Run in foreground (watch progress)
python3 train_main.py --config gpu_test --iterations 5 --yes

# Option 2: Run in background (can close terminal)
nohup python3 train_main.py --config gpu_test --iterations 5 --yes > training.log 2>&1 &

# Check progress if running in background:
tail -f training.log
```

**Expected output:**
```
ITERATION 1/5
  Self-play: Game 1/5 | Moves: 42 | Winner: DEFENDER | Time: 8.2s
  Self-play: Game 2/5 | Moves: 38 | Winner: ATTACKER | Time: 7.1s
  ...
  Training network... Loss: 2.345
  Evaluation: 12/20 wins (60.0%)
  ✓ New model accepted!

ITERATION 2/5
  ...
```

**Timeline:**
- Each iteration: ~10-15 minutes
- Total time: ~50-75 minutes
- Total cost: ~$0.50-1.00

## Step 6: Monitor Progress

While training runs:

```bash
# Check GPU usage (should be ~80-100%)
nvidia-smi

# Check disk space
df -h

# View training progress
tail -f training.log  # if running in background

# Check for checkpoints
ls -lh checkpoints_gpu_test/
```

## Step 7: Download Trained Model

After training completes:

```bash
# Check what was created
ls -lh checkpoints_gpu_test/

# You should see:
# checkpoint_iter_2.pt
# checkpoint_iter_4.pt
# best_model.pt
```

Download `best_model.pt`:

**Method A: Web file browser**
- Click "Connect" → "HTTP Service"
- Navigate to `checkpoints_gpu_test/`
- Download `best_model.pt`

**Method B: SCP**
```bash
# From your local machine:
scp -P XXXXX -i ~/.ssh/id_ed25519 \
  root@X.X.X.X:~/hnefatafl/checkpoints_gpu_test/best_model.pt \
  ~/Downloads/
```

## Step 8: Test Your Model Locally

Copy the model to your local checkpoints directory:

```bash
# On your local machine:
cd ~/Dokumente/Coding/Projects/copenhagen-hnefatafl
mkdir -p checkpoints
cp ~/Downloads/best_model.pt checkpoints/

# Play against it!
./venv/bin/python play.py --game-mode cpu
```

## Step 9: Stop RunPod Instance

**IMPORTANT**: Don't forget to stop the pod!

1. Go to RunPod dashboard
2. Find your pod
3. Click "Stop" or "Terminate"

**Verify charges**: Check "Billing" → Should be ~$0.50-1.00

## Troubleshooting

### "CUDA not available"
- Wrong template: Make sure you selected a GPU pod (not CPU)
- Driver issue: Try running `nvidia-smi` - if it fails, restart the pod

### "Out of memory"
Your GPU is too small. Either:
- Use smaller config: `--config quick` (but much slower)
- Or rent a GPU with more VRAM (RTX 3060 12GB minimum)

### Training seems stuck
- Check GPU usage: `nvidia-smi` (should be 80-100%)
- Check if process is running: `ps aux | grep train_main`
- Look at logs: `tail -f training.log`

### Process died
- Check logs: `cat training.log`
- Usually means OOM (out of memory) - need bigger GPU

## After 5 Iterations

You'll have a **very basic** prototype that:
- Understands basic rules
- Makes semi-random moves
- Is NOT competitive yet

**Next steps:**
1. Test it locally: `./venv/bin/python play.py --game-mode cpu`
2. If happy, run longer training (50-100 iterations) for ~$5-10
3. Use cheaper Vast.ai for the longer run

## Cost Summary

- **5 iterations**: $0.50-1.00 (1-2 hours)
- **50 iterations**: $5-7 (10-15 hours)
- **100 iterations**: $10-14 (20-30 hours)

Your $10 RunPod credit can get you:
- 5 iterations (test) + 50-70 iterations (decent model)

## Questions?

- Check GPU_RENTAL_GUIDE.md for more details
- Check QUICK_REFERENCE.md for post-training tips
- RunPod support: https://www.runpod.io/discord
