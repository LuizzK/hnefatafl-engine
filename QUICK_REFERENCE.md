# Quick Reference - After MVP Training

## Is Training Still Running?

```bash
ps aux | grep train_main.py | grep -v grep
```
- **Shows process** → Still running ✓
- **Empty** → Finished!

## Check Progress

```bash
# View log (may be empty until end)
tail -50 mvp_training.log

# Check for checkpoints
ls checkpoints_mvp/
```

## When Training Completes

### ✅ Verify Success

```bash
# 1. Check training log
cat mvp_training.log | tail -20
# Look for: "TRAINING COMPLETE!"

# 2. Count iterations
grep "ITERATION.*COMPLETE" mvp_training.log | wc -l
# Should be: 5

# 3. Check checkpoints exist
ls -lh checkpoints_mvp/
# Should see: checkpoint_iter_2.pt, checkpoint_iter_4.pt, best_model.pt

# 4. Check for errors
grep -i "error\|nan\|exception" mvp_training.log
# Should be empty (or minor warnings only)

# 5. Test model loads
./venv/bin/python -c "
import torch
checkpoint = torch.load('checkpoints_mvp/best_model.pt', map_location='cpu')
print('✓ Model loads OK')
print(f'✓ Iterations: {checkpoint[\"iteration\"]}')
print(f'✓ Games: {checkpoint[\"total_games\"]}')
"
```

### If All Pass → Ready for GPU! 🎉

---

## Next Steps After Success

### 1. Read Documentation
```bash
cat GPU_RENTAL_GUIDE.md
```

### 2. Choose GPU Platform
- **Vast.ai**: Cheapest, good for budget ($0.15-0.25/hr)
- **RunPod**: Best balance, reliable ($0.34/hr)

### 3. Decide Budget
- **$20-30**: 500 iterations (competent)
- **$40-60**: 1000 iterations (strong)
- **$80-120**: 2000 iterations (expert)

### 4. Start Small
```bash
# On GPU (after setup):
python train_main.py --config standard --iterations 10 --yes
# Cost: ~$0.50-1, verifies GPU works
```

### 5. Full Training
```bash
# After small test succeeds:
python train_main.py --config standard --iterations 500 --yes
# Cost: ~$20-30 for competent engine
```

---

## If Training Failed

### Check Error
```bash
# View full log
cat mvp_training.log

# Find error message
grep -A 10 -i "error\|exception" mvp_training.log
```

### Common Issues

**1. Out of Memory**
- CPU ran out of RAM
- Usually won't happen with MVP config
- If it does: Close other programs and retry

**2. NaN in Loss**
- Would indicate bug in training code
- Very unlikely with MVP config
- Report to developer

**3. Process Killed**
- System ran out of memory
- Laptop went to sleep
- Just restart: `./venv/bin/python train_main.py --config mvp --iterations 5 --yes`

---

## Can I Close Things?

### ✅ Can Close:
- Claude Code (training continues)
- Browser
- Other applications
- Terminal (if you started training in background)

### ❌ Don't:
- Shut down laptop
- Let laptop run out of battery
- Kill the Python process manually

### ⚠️ Laptop Sleep:
- Training will PAUSE when laptop sleeps
- Resumes when you wake it
- Best to disable sleep for training duration:
  ```bash
  # Keep laptop awake during training (Ubuntu/Debian)
  systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

  # Re-enable sleep after training
  systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
  ```

---

## Monitoring Commands

```bash
# CPU usage (training should be 100%+)
top -b -n 1 | grep python

# Memory usage
free -h

# How long has it been running?
ps -p $(pgrep -f train_main.py) -o etime=

# Watch log file grow (Ctrl+C to exit)
watch -n 5 'ls -lh mvp_training.log'
```

---

## Time Estimates

- **Iteration 1**: 5-10 min (longest, generating first games)
- **Iterations 2-5**: 4-8 min each (faster with replay buffer)
- **Total**: 30-60 minutes

---

## After GPU Training Completes

### Play Against Your Engine!

```bash
# Human vs trained AI
./venv/bin/python play.py --game-mode cpu

# The GUI automatically loads checkpoints/best_model.pt
```

### Test Engine Strength

```bash
# Let engine play against itself
./venv/bin/python -c "
from hnefatafl.game import HnefataflGame
from hnefatafl.mcts import MCTS
from hnefatafl.network import create_model
import torch

# Load trained model
model = create_model(num_channels=128, num_res_blocks=10, device='cpu')
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Play a game
mcts = MCTS(neural_network=model, num_simulations=800)
game = HnefataflGame()

while not game.is_game_over():
    move, _ = mcts.search(game, temperature=0.0)
    game.make_move(move)
    print(f'Move {game.move_count}: {move}')

print(f'Winner: {game.get_winner()}')
"
```

---

## Emergency: Stop Training

```bash
# Find process ID
ps aux | grep train_main.py | grep -v grep

# Kill it (replace PID with actual process ID)
kill <PID>

# Or kill all training processes
pkill -f train_main.py
```

**Note:** Training saves checkpoints every 2 iterations, so you'll lose at most 2 iterations of work (~10-15 min).

---

## Questions?

- Check **GPU_RENTAL_GUIDE.md** for GPU training
- Check **PRODUCTION_READY.md** for complete overview
- Check **QUICK_START.md** for how to play
- Training log will have all output when complete

---

**Current Status File:** MVP_TRAINING_STATUS.md

**Expected Completion:** 30-60 minutes from start time
