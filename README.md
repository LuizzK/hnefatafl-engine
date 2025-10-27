# Copenhagen Hnefatafl Engine

An AlphaZero-style engine for Copenhagen Hnefatafl using self-play reinforcement learning.

## Features

- Complete implementation of Copenhagen Hnefatafl rules (11x11 board)
- Special rules: Shieldwall Capture, Edge Forts, King capture mechanics
- Neural network-based engine (no manual heuristics)
- Monte Carlo Tree Search (MCTS) for move selection
- Self-play training pipeline
- Simple GUI for human vs engine play

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### GUI Mode (Play against the engine)
```bash
python main.py --mode gui
```

### Training Mode (Self-play learning)
```bash
python main.py --mode train
```

### CLI Mode
```bash
python main.py --mode play
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

## Training

The engine learns entirely through self-play without manual heuristics, similar to AlphaZero.
Training typically requires several days to reach strong play levels.
