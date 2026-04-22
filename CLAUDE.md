# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Copenhagen Hnefatafl AlphaZero engine - a neural network-based game engine that learns entirely through self-play reinforcement learning (no manual heuristics). Implements complete Copenhagen Hnefatafl rules on 11x11 board including standard piece movement, shieldwall capture, edge forts, and all win conditions.

## Essential Commands

### Training
```bash
# Test the pipeline (30-60 min on CPU)
./venv/bin/python train_main.py --config mvp --iterations 5 --yes

# Quick CPU test (2-3 hours)
./venv/bin/python train_main.py --config quick --iterations 5

# Standard GPU training (requires CUDA)
./venv/bin/python train_main.py --config standard --iterations 1000

# Show all available training configurations
./venv/bin/python train_main.py --show-configs

# Resume from checkpoint
./venv/bin/python train_main.py --resume checkpoints/checkpoint_iter_50.pt
```

### Playing
```bash
# Human vs Human GUI
./venv/bin/python play.py

# Human vs CPU (play against AI)
./venv/bin/python play.py --game-mode cpu

# Setup mode (custom positions)
./venv/bin/python play.py --game-mode setup

# CLI mode
./venv/bin/python play.py --mode cli
```

### Testing
```bash
# Run all tests (35 tests should pass)
./venv/bin/pytest tests/ -v

# Run specific test file
./venv/bin/pytest tests/test_game.py -v

# Run tests with output
./venv/bin/pytest tests/ -v -s
```

### Validation Scripts
```bash
# Validate before training
./venv/bin/python validate_before_training.py

# Verify MCTS implementation
./venv/bin/python verify_mcts.py

# Verify neural network
./venv/bin/python verify_network.py

# Test move encoding
./venv/bin/python test_move_encoding.py

# Test MCTS policy
./venv/bin/python test_mcts_policy.py
```

## Architecture

### Training Pipeline Flow
1. **Self-play** (hnefatafl/selfplay.py): Model plays games against itself using MCTS, generates training positions
2. **Training** (hnefatafl/train.py): Neural network learns from positions, updates weights
3. **Evaluation** (hnefatafl/train.py): New model plays against previous best, replaces if win rate > 55%
4. **Iteration**: Repeat for 100s-1000s of iterations until model reaches desired strength

### Core Components

**hnefatafl/game.py** - Game engine and rules
- `HnefataflGame` class: Complete game state and rule implementation
- Board representation: 11x11 numpy array with piece types (EMPTY=0, ATTACKER=1, DEFENDER=2, KING=3)
- Move encoding: 4840-dim vector indexed as `from_square * 40 + direction * 10 + (distance - 1)` where direction ∈ {0=up,1=down,2=left,3=right} and distance ∈ [1,10]. Pieces move only orthogonally (rook-like).
- Special rules: Shieldwall capture, edge forts (4+ defenders on edge = win), king capture (4-sided or 3-sided near throne)
- Win detection: King reaches corner, attackers capture king, edge fort formation, or move repetition/stalemate

**hnefatafl/network.py** - Neural network architecture
- ResNet-style CNN with policy and value heads (AlphaZero architecture)
- Input: 15-channel 11x11 board representation (piece positions, legal moves, historical features)
- Output: Policy head (move probabilities), Value head (position evaluation -1 to 1)
- Configurable size: num_channels (32-256), num_res_blocks (2-20)
- Policy output: 4840-dim vector (121 squares × 4 directions × 10 distances); illegal-move indices are pruned during MCTS expansion (priors only assigned to legal moves, then renormalized)

**hnefatafl/mcts.py** - Monte Carlo Tree Search
- PUCT algorithm: UCB = Q + c_puct * P * sqrt(parent_N) / (1 + N)
- Neural network guides search: Policy provides prior probabilities, Value provides leaf evaluation
- Temperature parameter controls exploration: T=1.0 for early game (move_num < threshold), T=0.0 for late game (greedy)
- Dirichlet noise added to root for exploration during training: alpha=0.3, epsilon=0.25
- Returns: (best_move, visit_count_distribution) for training

**hnefatafl/selfplay.py** - Self-play game generation
- Plays full games using MCTS with current model
- Generates training data: (board_state, policy_target, value_target) for each position
- Policy target: MCTS visit count distribution (NOT raw network output)
- Value target: Game outcome from current player's perspective (1=win, -1=loss, 0=draw)
- Max move limit prevents infinite games (attackers win on timeout, simulates numerical superiority)

**hnefatafl/train.py** - Training loop coordinator
- `Trainer` class orchestrates entire training pipeline
- Replay buffer: Stores recent game positions (50K-1M positions), samples batches for training
- Loss function: policy_loss (cross-entropy) + value_loss (MSE) + L2 regularization
- Optimizer: Adam with learning rate decay (lr_decay_steps, lr_decay_gamma)
- Checkpointing: Saves model every N iterations, keeps best model based on evaluation
- Evaluation: New model vs old model in 20-100 games, win rate must exceed 55% to replace

**hnefatafl/gui.py** - Pygame GUI
- Drag-and-drop piece movement
- Undo functionality
- Hint system (shows MCTS-suggested moves)
- Multiple game modes: Human vs Human, Human vs CPU, Setup mode
- Auto-loads best_model.pt checkpoint if available

### Configuration System (config.py)

Five training configurations with different resource requirements:
- **MVPTrainingConfig**: CPU test (30-60 min, 2 games/iter, 25 sims, tiny model)
- **GPUTestConfig**: GPU verification (30-60 min, 5 games/iter, 800 sims, standard model)
- **QuickTrainingConfig**: CPU debugging (2-3 hours, 10 games/iter, 100 sims, small model)
- **StandardTrainingConfig**: Production GPU training (days-weeks, 100 games/iter, 800 sims, 128ch/10block model)
- **IntenseTrainingConfig**: High-end GPU (weeks, 500 games/iter, 1600 sims, 256ch/20block model)

Key parameters:
- `num_simulations`: MCTS depth (25-1600)
- `num_games_per_iteration`: Games per training iteration (2-500)
- `batch_size`: Training batch size (16-512)
- `replay_buffer_size`: Max positions to store (1K-1M)
- `num_channels`, `num_res_blocks`: Model capacity
- `device`: 'cpu' or 'cuda'

## Critical Implementation Details

### Move Encoding
- Moves represented as 4-tuple: (from_row, from_col, to_row, to_col)
- Encoded as `from_square * 40 + direction * 10 + (distance - 1)` where `from_square = from_row * 11 + from_col`, direction ∈ {0=up, 1=down, 2=left, 3=right}, distance ∈ [1, 10] — see `hnefatafl/game.py:encode_move`
- Total action space: 121 × 4 × 10 = 4840 possible moves (Copenhagen Hnefatafl uses rook-like movement only)
- Network outputs 4840-dim vector; MCTS picks priors via `encode_move(move)` index and renormalizes over legal moves (illegal-index mass is dropped)

### Dirichlet Noise
- Added to root priors at the start of every self-play MCTS search (`MCTS.search(..., add_noise=True)`)
- Defaults: `alpha=0.3`, `epsilon=0.25`
- **Never** applied during evaluation (`_play_evaluation_game` in train.py) — evaluation uses greedy visit-count selection with clean priors

### Board State Representation (15 channels)
1. Current player's pieces (binary)
2. Opponent's pieces (binary)
3. King position (binary)
4. Throne position (binary)
5. Corner positions (binary)
6. Legal move indicators (binary)
7-15. Historical board states (last 8 positions for repetition detection)

### Edge Fort Win Condition
- 4+ defenders on same edge forming connected line
- Must touch at least one corner
- If unbreakable (attackers can't capture), defenders win
- Implementation: hnefatafl/game.py:_check_edge_fort()

### Training Timeline Expectations
- Random → Coherent: 4-8 hours (learns basic rules)
- Coherent → Competent: 1-2 days (understands tactics)
- Competent → Strong: 3-7 days (advanced strategy)
- Strong → Master: 1-2 weeks (master-level play)

### Device Compatibility
- CPU training: Works but 10-50x slower than GPU
- CUDA required for GPU: torch.cuda.is_available() must be True
- RTX 5090 requires PyTorch 2.7+ for sm_120 CUDA architecture
- Use --device cpu to force CPU even if CUDA available

## Workflow Guidelines

### Before Making Changes
1. Read relevant documentation (PRODUCTION_READY.md, EDGE_FORT_IMPLEMENTATION.md, WHATS_MISSING.md)
2. Run tests to ensure current state is working: `./venv/bin/pytest tests/ -v`
3. For training changes, test with MVP config first before GPU training

### Testing Changes
1. Unit tests must pass: `./venv/bin/pytest tests/ -v`
2. For training code changes: Run MVP config (5 iterations) to verify pipeline works
3. For game logic changes: Run specific game tests and play a few moves in GUI

### Adding New Features
1. Game rules: Add to hnefatafl/game.py, write tests in tests/test_game.py
2. Network architecture: Modify hnefatafl/network.py, verify with verify_network.py
3. MCTS changes: Modify hnefatafl/mcts.py, verify with verify_mcts.py
4. Training changes: Modify hnefatafl/train.py or config.py, test with MVP config

### Debugging Training Issues
- Check logs in checkpoint directory
- Monitor loss values (policy loss + value loss should decrease)
- Verify replay buffer not empty: Should grow to buffer_size
- Check evaluation win rates: Should be around 50% if model improving steadily
- Look for NaN values: Indicates numerical instability (very rare)

## Common Gotchas

1. **Always use venv Python**: Use `./venv/bin/python` not just `python`
2. **GPU memory**: Standard config needs ~4GB VRAM, Intense needs ~8GB
3. **Move encoding**: Must mask illegal moves before MCTS, otherwise policy will suggest invalid moves
4. **MCTS temperature**: Must use T=1.0 during training for exploration, T=0.0 for best play
5. **Value targets**: Must be from current player's perspective (1=win, -1=loss), not absolute
6. **Checkpoints**: best_model.pt is the latest accepted model, checkpoint_iter_N.pt are iteration snapshots
7. **Edge fort detection**: Only counts if defenders form unbreakable formation (see _check_edge_fort)

## File Organization

Core implementation:
- `hnefatafl/` - Main package (game, network, mcts, selfplay, train, gui)
- `config.py` - Training configurations
- `train_main.py` - Training entry point
- `play.py` - GUI/CLI entry point

Tests:
- `tests/test_game.py` - Game logic tests
- `tests/test_edge_forts.py` - Edge fort rule tests
- `test_*.py` (root level) - Quick verification scripts

Documentation:
- `PRODUCTION_READY.md` - Complete guide (start here)
- `QUICK_START.md` - How to play
- `QUICK_REFERENCE.md` - Post-training reference
- `GPU_RENTAL_GUIDE.md` - GPU training setup
- `EDGE_FORT_IMPLEMENTATION.md` - Edge fort rules
- `WHATS_MISSING.md` - Historical (now complete)

Generated during training:
- `checkpoints/` - Model checkpoints (ignored by git)
- `checkpoints_mvp/`, `checkpoints_gpu_test/` - Config-specific checkpoints
- `*.log` - Training logs

## Important Notes

- System is production-ready, all 35 tests pass
- Always run MVP config first before expensive GPU training to catch bugs
- Training requires significant compute: Use GPU for real training, CPU only for testing
- Model learns entirely from self-play, no manual heuristics or opening books
- First 50-100 iterations will produce weak play while model learns basic rules
