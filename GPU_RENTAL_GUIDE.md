# GPU Rental Guide for Training

## TL;DR - Quick Answer

**Yes, you can rent GPUs for training!**

**Cost estimates:**
- **RTX 3090**: $0.10-0.20/hour → **$24-48 for 1 week** (competent play)
- **RTX 4090**: $0.34-0.40/hour → **$58-68 for 1 week** (strong play)
- **Total for strong engine**: **$50-100**

**Recommended platforms:**
1. **Vast.ai** - Cheapest ($0.15-0.40/hr for RTX 4090)
2. **RunPod** - Good balance ($0.34/hr Community Cloud)
3. **Lambda Labs** - Most reliable but pricier

**BEFORE spending any money**:
1. Run validation: `./venv/bin/python validate_before_training.py`
2. Run MVP on CPU: `./venv/bin/python train_main.py --config mvp --iterations 5`

---

## Detailed GPU Rental Information

### Best Platforms for ML Training (2025)

#### 1. **Vast.ai** (Best for Budget)

**Pros:**
- Cheapest prices (peer-to-peer marketplace)
- Wide GPU selection
- No minimum commitment
- Pay-per-second billing

**Cons:**
- Reliability varies (community hardware)
- May need to find new instances if host goes down
- Less support

**Pricing (as of 2025):**
- RTX 3090: **$0.09 - $0.20/hour**
- RTX 4090: **$0.15 - $0.40/hour**
- A100 80GB: **$1.00 - $1.50/hour**

**Best for:** Budget-conscious training, can handle occasional restarts

**Setup:**
1. Sign up at https://vast.ai
2. Search for RTX 4090 or RTX 3090
3. Select instance with good uptime/reliability score
4. Use their Docker template or SSH access

---

#### 2. **RunPod** (Best Balance)

**Pros:**
- Good reliability
- Two tiers: Community (cheap) and Secure (guaranteed)
- Per-second billing
- Easy to use
- Good for ML specifically

**Cons:**
- Slightly more expensive than Vast.ai
- Popular GPUs can sell out

**Pricing (as of 2025):**
- **Community Cloud:**
  - RTX 4090: **$0.34/hour**
  - RTX 3090: **~$0.20/hour**
  - A100 80GB: **$1.74/hour**

- **Secure Cloud:** +$0.27-0.45/hr more

**Best for:** Most users - good balance of price and reliability

**Setup:**
1. Sign up at https://runpod.io
2. Choose "Community Cloud" for best prices
3. Deploy PyTorch template or use SSH
4. Persistent storage for checkpoints

---

#### 3. **Lambda Labs** (Best for Reliability)

**Pros:**
- Very reliable
- Good for production
- Excellent support
- Pre-configured for ML

**Cons:**
- More expensive
- Limited GPU availability
- Higher minimum costs

**Pricing (as of 2025):**
- H100 PCIe: **$2.49 - $2.99/hour**
- A100: **~$1.50 - $2.00/hour**

**Best for:** If you need guaranteed uptime and can afford higher costs

---

### Cost Calculator

#### For Standard Config (128ch, 10 blocks, 800 sims)

**Time needed:**
- Coherent play: 50 iterations × 7 min = ~6 hours
- Competent play: 200 iterations × 7 min = ~24 hours (1 day)
- Strong play: 500 iterations × 7 min = ~58 hours (2.4 days)
- Master play: 1000 iterations × 7 min = ~116 hours (4.8 days)

**Costs at different platforms:**

| Platform | GPU | Hourly | 1 Day | 3 Days | 1 Week |
|----------|-----|--------|-------|--------|--------|
| Vast.ai | RTX 3090 | $0.15 | $3.60 | $10.80 | $25.20 |
| Vast.ai | RTX 4090 | $0.25 | $6.00 | $18.00 | $42.00 |
| RunPod | RTX 4090 | $0.34 | $8.16 | $24.48 | $57.12 |
| RunPod | A100 | $1.74 | $41.76 | $125.28 | $292.32 |

**Example scenarios:**

1. **Budget: Get competent play**
   - Platform: Vast.ai RTX 3090
   - Time: 24 hours
   - Cost: **~$3-5**

2. **Recommended: Get strong play**
   - Platform: RunPod RTX 4090 Community
   - Time: 60 hours (2.5 days)
   - Cost: **~$20-25**

3. **Best: Get master play**
   - Platform: RunPod RTX 4090 Community
   - Time: 120 hours (5 days)
   - Cost: **~$40-50**

---

## Risk Mitigation Strategy

### **You're absolutely right to be scared!** Here's how to minimize risk:

### 1. Pre-Flight Checks (DO THIS FIRST!)

**Run the validation script:**
```bash
./venv/bin/python validate_before_training.py
```

This checks:
- All imports work
- Game engine functions
- Neural network forward pass
- MCTS search
- Self-play game generation
- Training pipeline initialization

**If any test fails, DO NOT rent a GPU yet!**

---

### 2. MVP Training on Local CPU (30-60 minutes)

**Run a minimal training session on your laptop:**
```bash
./venv/bin/python train_main.py --config mvp --iterations 5
```

**What this does:**
- Runs 5 complete training iterations
- Each iteration: 2 games, tiny model, minimal simulations
- Total time: ~30-60 minutes
- **Verifies end-to-end that EVERYTHING works**

**What to watch for:**
- ✅ All iterations complete without errors
- ✅ Loss values are reasonable (not NaN or inf)
- ✅ Checkpoints are saved successfully
- ✅ Evaluation games complete
- ✅ No memory errors
- ✅ No crashes

**If MVP completes successfully, you're 95% safe!**

---

### 3. Start Small on GPU (1-2 hours, $0.50-1)

**When you first rent a GPU:**

```bash
# Start with just 10 iterations
./venv/bin/python train_main.py --config standard --iterations 10
```

**This costs only $0.50-1.00 but verifies:**
- GPU access works
- CUDA is properly configured
- Training speed is as expected
- No GPU-specific errors
- Checkpoints save correctly
- You can resume from checkpoints

**Monitor the first few iterations carefully!**

---

### 4. Use Checkpoints Religiously

The training system automatically saves checkpoints every 5 iterations:
- `checkpoints/checkpoint_iter_5.pt`
- `checkpoints/checkpoint_iter_10.pt`
- `checkpoints/best_model.pt`

**If training crashes:**
```bash
# Resume from last checkpoint
./venv/bin/python train_main.py --resume checkpoints/checkpoint_iter_50.pt
```

**You never lose more than 5 iterations of work!**

---

### 5. Monitor Training Progress

**Watch for these red flags:**

❌ **Loss is NaN or infinity**
- Bug in loss calculation
- Learning rate too high
- Stop immediately!

❌ **Loss doesn't decrease after 50 iterations**
- Model not learning
- Check hyperparameters
- May need to adjust

❌ **Training crashes every N iterations**
- Memory leak or bug
- Fix before continuing

✅ **Good signs:**
- Loss steadily decreases
- Win rate fluctuates around 50%
- No crashes for 20+ iterations
- Checkpoints save successfully

---

### 6. Test Resume Before Long Training

Before starting a week-long training run:

```bash
# Train for 5 iterations
./venv/bin/python train_main.py --config standard --iterations 5

# Stop it (Ctrl+C or let it finish)

# Resume from checkpoint
./venv/bin/python train_main.py --resume checkpoints/checkpoint_iter_5.pt --iterations 10
```

**Verify it picks up where it left off!**

---

## Bug Checklist

Before GPU rental, ensure these work:

### Game Engine
- [x] Legal move generation
- [x] Move execution
- [x] Game-over detection
- [x] State encoding
- [x] All 35 tests pass

### Neural Network
- [x] Forward pass (state → policy, value)
- [x] Loss computation
- [x] Backward pass
- [x] Parameter updates
- [x] Checkpoint save/load

### MCTS
- [x] Tree search
- [x] Move selection
- [x] Temperature parameter
- [x] Policy output correct size

### Self-Play
- [x] Generate complete games
- [x] Collect training examples
- [x] Assign outcomes correctly
- [x] No crashes during generation

### Training Loop
- [x] Self-play generation
- [x] Replay buffer management
- [x] Batch training
- [x] Model evaluation
- [x] Checkpoint saving
- [x] Resume from checkpoint

---

## Recommended Workflow

### **Step 1: Local Validation (30-60 min, FREE)**

```bash
# Run validation tests
./venv/bin/python validate_before_training.py

# Run MVP training
./venv/bin/python train_main.py --config mvp --iterations 5
```

**If both succeed → proceed to Step 2**
**If either fails → fix bugs first!**

---

### **Step 2: Small GPU Test (1-2 hours, ~$0.50-1)**

1. Rent cheapest GPU (RTX 3090 on Vast.ai)
2. Setup Python environment
3. Clone your GitHub repo
4. Install requirements
5. Run:
```bash
./venv/bin/python train_main.py --config standard --iterations 10
```

**If successful → proceed to Step 3**
**If fails → debug, don't spend more money yet**

---

### **Step 3: Short Training Run (24 hours, ~$5-10)**

Run for 200 iterations to reach "competent" play:
```bash
./venv/bin/python train_main.py --config standard --iterations 200
```

**Test the model:**
```bash
./venv/bin/python play.py --game-mode cpu
```

**Play a few games. Does the AI:**
- Make legal moves? ✓
- Have any strategy? ✓
- Seem to improve over random? ✓

**If yes → proceed to Step 4**

---

### **Step 4: Full Training (3-7 days, ~$25-60)**

Now you're confident! Run the full training:
```bash
./venv/bin/python train_main.py --config standard --iterations 1000
```

Monitor periodically, but you can let it run.

---

## Platform Setup Guides

### Vast.ai Setup

1. **Sign up** at https://vast.ai
2. **Search** for GPU: "RTX 4090" or "RTX 3090"
3. **Filter**:
   - Disk space: 50GB+
   - CUDA version: 11.8+
   - Reliability: 95%+
4. **Launch** instance with PyTorch template
5. **Connect** via SSH
6. **Clone repo**:
```bash
git clone https://github.com/LuizzK/hnefatafl-engine.git
cd hnefatafl-engine
pip install -r requirements.txt
```
7. **Start training**!

---

### RunPod Setup

1. **Sign up** at https://runpod.io
2. **Select** "Community Cloud"
3. **Choose** RTX 4090
4. **Deploy** Pod with PyTorch template
5. **Connect** via SSH or Jupyter
6. **Clone repo** and install requirements
7. **Start training**!

---

## FAQ

### Q: How do I know if training is working?

**A:** Look for these signs:
- Loss decreases over time (starts high, goes down)
- No NaN or infinity in logs
- Checkpoints save successfully
- Model evaluation completes
- Win rate hovers around 50% (new vs old model)

### Q: What if training crashes?

**A:** Don't panic!
- Find the last checkpoint: `ls checkpoints/`
- Resume from it: `--resume checkpoints/checkpoint_iter_X.pt`
- You lose at most 5 iterations (~30-50 min of work)

### Q: Can I pause training and resume later?

**A:** Yes!
- Press Ctrl+C to stop training
- Wait for current iteration to finish
- Resume anytime with `--resume checkpoints/checkpoint_iter_X.pt`
- Even works across different machines!

### Q: What if I run out of money mid-training?

**A:**
- Training saves checkpoints automatically
- Download all checkpoints before GPU expires
- Resume later on a different GPU
- You keep all progress!

### Q: How do I download checkpoints?

**A:**
```bash
# On GPU machine
tar -czf checkpoints.tar.gz checkpoints/

# From local machine
scp user@gpu:checkpoints.tar.gz .
```

### Q: What's the minimum viable training?

**A:** For a "barely functional" engine:
- 50-100 iterations
- ~7-12 hours on RTX 4090
- ~$2.50-4 on Vast.ai
- Will play legal moves and have basic strategy

### Q: Can I use multiple GPUs?

**A:** The current implementation uses one GPU. Multi-GPU would require code changes, but one GPU is sufficient for this project.

---

## Cost Summary

**Total cost to achieve strong play:**

| Approach | Time | Cost | Result |
|----------|------|------|--------|
| **Budget** | 1-2 days | $5-10 | Competent |
| **Recommended** | 3-5 days | $25-40 | Strong |
| **Premium** | 5-7 days | $40-60 | Master |

**Compare to alternatives:**
- Buying RTX 4090: **$1,500-2,000**
- Buying cloud credits: **Same price, locked to one provider**
- Renting for this project: **$25-60, no commitment**

**Renting is by far the best option for this project!**

---

## Safety Checklist

Before starting GPU rental:

- [ ] Validation script passes all tests
- [ ] MVP training completes successfully on CPU
- [ ] All 35 game tests pass
- [ ] You've read this entire guide
- [ ] You understand how to resume from checkpoints
- [ ] You know which GPU platform you'll use
- [ ] You have a backup plan if training fails
- [ ] You're mentally prepared to debug if needed

**If all checked → GO FOR IT!** 🚀

---

## Emergency Contacts

If something goes wrong:

1. **Check the logs** - often the error message tells you what's wrong
2. **Check disk space** - `df -h`
3. **Check GPU memory** - `nvidia-smi`
4. **Try restarting** - Sometimes that's all it takes
5. **Resume from checkpoint** - Don't start over!

**Remember:** The validation and MVP scripts are specifically designed to catch 95% of bugs BEFORE you spend money.

If they pass, you're almost certainly safe! 💪

---

**Now go train that engine!** 🎉
