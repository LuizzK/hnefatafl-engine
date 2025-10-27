# Copenhagen Hnefatafl Engine

An AlphaZero-style engine for Copenhagen Hnefatafl using self-play reinforcement learning.

## 🎉 Status: PRODUCTION READY!

**All components are complete! Ready for training!**

See **[PRODUCTION_READY.md](PRODUCTION_READY.md)** for the complete guide.

## Features

- ✅ Complete implementation of Copenhagen Hnefatafl rules (11x11 board)
- ✅ Special rules: Shieldwall Capture, Edge Forts, King capture mechanics
- ✅ Neural network-based engine (no manual heuristics)
- ✅ Monte Carlo Tree Search (MCTS) for move selection
- ✅ Self-play training pipeline
- ✅ Enhanced GUI with undo, hints, multiple modes, drag & drop
- ✅ Complete training system with checkpoints and evaluation
- ✅ Three training configurations (Quick/Standard/Intense)
- ✅ 35 tests passing

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Start Training

```bash
# Test the pipeline first (2-3 hours on CPU)
./venv/bin/python train_main.py --config quick --iterations 5

# Start real training (requires GPU, 3-7 days)
./venv/bin/python train_main.py --config standard --iterations 1000

# Show training options
./venv/bin/python train_main.py --show-configs
```

### 2. Play the Game

```bash
# Human vs Human (GUI)
./venv/bin/python play.py

# Human vs CPU (play against AI)
./venv/bin/python play.py --game-mode cpu

# Setup custom positions
./venv/bin/python play.py --game-mode setup

# CLI mode
./venv/bin/python play.py --mode cli
```

### 3. Run Tests

```bash
./venv/bin/pytest tests/ -v
```

## Project Structure

- `hnefatafl/game.py` - Game logic and rules
- `hnefatafl/network.py` - Neural network architecture
- `hnefatafl/mcts.py` - Monte Carlo Tree Search
- `hnefatafl/selfplay.py` - Self-play game generation
- `hnefatafl/train.py` - Training pipeline
- `hnefatafl/gui.py` - Pygame interface
- `tests/` - Unit tests

## Copenhagen Hnefatafl Rules

- 11x11 board with 1 king + 12 defenders vs 24 attackers
- All pieces move like rooks in chess
- Standard capture: Sandwich opponent between two pieces
- Shieldwall capture: Capture multiple pieces along edge
- Edge forts: Defenders can win by creating unbreakable edge formation
- King wins by reaching any corner square
- Attackers win by capturing the king (4-sided surround, 3-sided near throne)

## Documentation

- **[PRODUCTION_READY.md](PRODUCTION_READY.md)** - Complete guide (start here!)
- **[QUICK_START.md](QUICK_START.md)** - How to play the game
- **[EDGE_FORT_IMPLEMENTATION.md](EDGE_FORT_IMPLEMENTATION.md)** - Edge fort rules
- **[WHATS_MISSING.md](WHATS_MISSING.md)** - What was missing (now complete!)

## Training

The engine learns entirely through self-play without manual heuristics, similar to AlphaZero.

**Expected timeline:**
- Random → Coherent: 4-8 hours (learns basic rules)
- Coherent → Competent: 1-2 days (understands tactics)
- Competent → Strong: 3-7 days (advanced strategy)
- Strong → Master: 1-2 weeks (master-level play)

See **[PRODUCTION_READY.md](PRODUCTION_READY.md)** for detailed training instructions.
