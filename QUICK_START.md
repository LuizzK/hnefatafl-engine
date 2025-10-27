# Quick Start Guide - Copenhagen Hnefatafl

## Play the Game NOW! 🎮

### GUI Mode (Visual, Click to Move)
```bash
./venv/bin/python play.py
# OR
./venv/bin/python play.py --mode gui
```

**How to play:**
- **Click a piece** to select it (your color)
- **Legal moves** will be highlighted in blue
- **Click destination** to move
- **Click ESC** to quit
- **Press R** to restart

**Pieces:**
- **Black circles** = Attackers (move first)
- **White circles** = Defenders
- **White with gold border** = King

**Special squares:**
- **Gold square** = Throne (center)
- **Red squares** = Corners (king's escape squares)

### CLI Mode (Text-based, Type Moves)
```bash
./venv/bin/python play.py --mode cli
```

**Commands:**
- `f6-f8` = Move from f6 to f8
- `moves` = Show all legal moves
- `help` = Show instructions
- `quit` = Exit game

---

## Game Rules (Quick Reference)

### Objective
- **Defenders**: Get king to any corner OR form unbreakable edge fort
- **Attackers**: Capture the king

### Movement
- All pieces move like rooks (straight lines, any distance)
- Cannot jump over pieces
- Only king can land on throne/corners

### Capture
- **Standard**: Sandwich enemy between two pieces
- **Shieldwall**: Capture multiple pieces along edge by bracketing both ends
- **King**: Requires 4-sided surround (3-sided if next to throne)
- **King is immune on board edge!**

### Special Rules
- Throne and corners count as hostile squares for captures
- Pieces can pass through empty throne
- King cannot be captured on edge
- Repetition = attacker win

---

## Examples

### Starting Position
```
11|X . . A A A A A . . X |11
10|. . . . . A . . . . . |10
 9|. . . . . . . . . . . |9
 8|A . . . . D . . . . A |8
 7|A . . . D D D . . . A |7
 6|A A . D D K D D . A A |6
 5|A . . . D D D . . . A |5
 4|A . . . . D . . . . A |4
 3|. . . . . . . . . . . |3
 2|. . . . . A . . . . . |2
 1|X . . A A A A A . . X |1
```

**Legend:**
- `A` = Attacker (black)
- `D` = Defender (white)
- `K` = King (white with crown)
- `X` = Corner
- `T` = Throne

### Example Moves
1. Attacker: `f2-f4` (advance toward center)
2. Defender: `f6-f8` (king moves up)
3. Attacker: `a4-d4` (move along rank)
4. Defender: `f8-g8` (king moves toward corner)

---

## All Copenhagen Rules Implemented ✅

- ✅ **11×11 board** with correct starting position
- ✅ **Standard capture** (sandwich)
- ✅ **Shieldwall capture** (multi-piece edge captures)
- ✅ **King capture rules** (4-sided, 3-sided near throne, immune on edge)
- ✅ **Edge forts** (defender win when king forms unbreakable edge formation)
- ✅ **Corner escape** (king reaching corner wins)
- ✅ **Throne rules** (hostile to attackers, pass-through when empty)
- ✅ **Repetition detection** (3-fold = attacker win)

---

## Tips

### For Attackers
- Surround the king early
- Use shieldwall to break edge formations
- Control center and corners
- Force king away from edges

### For Defenders
- Rush king to nearest corner
- Form protective formations
- Use edge forts as backup
- Sacrifice defenders to create paths

---

## Quick Commands

```bash
# Play with GUI (default)
./venv/bin/python play.py

# Play with CLI
./venv/bin/python play.py --mode cli

# Run tests
./venv/bin/pytest tests/ -v

# Verify neural network
./venv/bin/python verify_network.py

# Verify MCTS
./venv/bin/python verify_mcts.py
```

---

Enjoy playing! 🎲👑
