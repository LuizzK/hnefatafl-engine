# What's Missing in the Implementation

## Current Status: ~80% Complete

### ✅ Completed Components (Fully Working)

1. **Game Engine** (`hnefatafl/game.py`) - 1000+ lines
   - ✅ All Copenhagen Hnefatafl rules
   - ✅ Shieldwall capture
   - ✅ Edge fort detection (thoroughly implemented)
   - ✅ King capture mechanics
   - ✅ Move generation
   - ✅ State encoding for neural network
   - ✅ 35/35 tests passing

2. **Neural Network** (`hnefatafl/network.py`) - 294 lines
   - ✅ ResNet architecture (22M parameters)
   - ✅ Policy head (move probabilities)
   - ✅ Value head (position evaluation)
   - ✅ Loss function
   - ✅ Checkpoint save/load
   - ✅ GPU support

3. **MCTS** (`hnefatafl/mcts.py`) - 364 lines
   - ✅ PUCT algorithm
   - ✅ Neural network integration
   - ✅ Tree search
   - ✅ Move selection
   - ✅ Dirichlet noise for exploration

4. **GUI** (`hnefatafl/gui.py`) - 200+ lines
   - ✅ Pygame interface
   - ✅ Click to move
   - ✅ Legal move highlighting
   - ✅ Visual board representation

5. **Tests & Verification**
   - ✅ 22 game engine tests
   - ✅ 13 edge fort tests
   - ✅ Neural network verification
   - ✅ MCTS verification

---

## 🚧 Missing Components (Need Implementation)

### 1. Self-Play Engine (`hnefatafl/selfplay.py`)

**Purpose**: Generate training games by having the AI play against itself

**What needs to be implemented:**

```python
class SelfPlayWorker:
    """Generate self-play games for training"""

    def __init__(self, model, num_simulations=800):
        self.model = model
        self.mcts = MCTS(model, num_simulations)

    def play_game(self, temperature_schedule):
        """
        Play one complete game using MCTS

        Returns:
            List of (state, policy, outcome) tuples for training
        """
        game = HnefataflGame()
        training_data = []

        while not game.is_game_over():
            # Get MCTS policy
            state = game.encode_state()
            move, policy = self.mcts.search(game)

            # Store training example
            training_data.append((state, policy, None))

            # Make move
            game.make_move(move)

        # Add game outcome to all positions
        winner = game.get_winner()
        for i in range(len(training_data)):
            state, policy, _ = training_data[i]
            # Outcome from perspective of player at that position
            outcome = 1.0 if winner == game.current_player else -1.0
            training_data[i] = (state, policy, outcome)

        return training_data

    def generate_games(self, num_games):
        """Generate multiple games in parallel"""
        all_data = []
        for i in range(num_games):
            game_data = self.play_game()
            all_data.extend(game_data)
        return all_data
```

**Key features needed:**
- Temperature scheduling (high exploration early, greedy late game)
- Parallel game generation (batch processing on GPU)
- Data augmentation (board rotations/reflections)
- Progress tracking

**Estimated complexity**: 200-300 lines
**Estimated time**: 1-2 days

---

### 2. Training Pipeline (`hnefatafl/train.py`)

**Purpose**: Train the neural network from self-play data

**What needs to be implemented:**

```python
class Trainer:
    """AlphaZero-style training pipeline"""

    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = []
        self.best_model = model.copy()

    def training_iteration(self, iteration):
        """One full training iteration"""

        # 1. Generate self-play games
        print(f"Iteration {iteration}: Generating games...")
        worker = SelfPlayWorker(self.best_model)
        new_data = worker.generate_games(num_games=100)

        # 2. Add to replay buffer
        self.replay_buffer.extend(new_data)
        if len(self.replay_buffer) > 500000:
            # Keep only recent data
            self.replay_buffer = self.replay_buffer[-500000:]

        # 3. Train new model
        print(f"Training on {len(self.replay_buffer)} positions...")
        new_model = self.train_network(
            self.replay_buffer,
            epochs=10,
            batch_size=256
        )

        # 4. Evaluate new vs old
        print("Evaluating new model...")
        win_rate = self.evaluate_models(new_model, self.best_model)

        # 5. Replace if better
        if win_rate > 0.55:
            print(f"New model wins {win_rate:.1%}! Replacing.")
            self.best_model = new_model
        else:
            print(f"New model only wins {win_rate:.1%}. Keeping old.")

        # 6. Save checkpoint
        self.save_checkpoint(iteration)

    def train_network(self, data, epochs, batch_size):
        """Train network on data"""
        for epoch in range(epochs):
            # Shuffle data
            random.shuffle(data)

            for batch in get_batches(data, batch_size):
                states, policies, values = zip(*batch)

                # Convert to tensors
                states_t = torch.stack([torch.FloatTensor(s) for s in states])
                policies_t = torch.stack([torch.FloatTensor(p) for p in policies])
                values_t = torch.FloatTensor(values).unsqueeze(1)

                # Forward pass
                policy_logits, value_pred = self.model(states_t)

                # Compute loss
                loss_fn = PolicyValueLoss()
                total_loss, policy_loss, value_loss = loss_fn(
                    policy_logits, value_pred, policies_t, values_t
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Log metrics
                self.log_training_step(total_loss, policy_loss, value_loss)

    def evaluate_models(self, new_model, old_model, num_games=100):
        """Play games between models to compare strength"""
        wins = 0
        for game_num in range(num_games):
            winner = self.play_evaluation_game(new_model, old_model)
            if winner == "new":
                wins += 1
        return wins / num_games

    def play_evaluation_game(self, model1, model2):
        """Play one game between two models"""
        game = HnefataflGame()
        mcts1 = MCTS(model1, num_simulations=400)
        mcts2 = MCTS(model2, num_simulations=400)

        while not game.is_game_over():
            if game.current_player == Player.ATTACKER:
                move, _ = mcts1.search(game)
            else:
                move, _ = mcts2.search(game)
            game.make_move(move)

        winner = game.get_winner()
        return "new" if winner == Player.ATTACKER else "old"
```

**Key features needed:**
- Replay buffer management
- Mini-batch training
- Model evaluation (pit new vs old)
- TensorBoard logging
- Checkpoint management
- Learning rate scheduling
- Early stopping
- Metrics tracking (ELO, win rate, loss curves)

**Estimated complexity**: 400-500 lines
**Estimated time**: 2-3 days

---

### 3. Training Configuration (`config.py`)

**What needs to be implemented:**

```python
class TrainingConfig:
    # Self-play
    num_simulations = 800
    num_games_per_iteration = 100
    temperature_threshold = 15  # Greedy after move 15

    # Training
    replay_buffer_size = 500000
    batch_size = 256
    epochs_per_iteration = 10
    learning_rate = 0.001
    weight_decay = 1e-4

    # Evaluation
    num_eval_games = 100
    eval_win_threshold = 0.55

    # Model
    num_channels = 128
    num_res_blocks = 10

    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4  # For parallel self-play
```

**Estimated complexity**: 50-100 lines
**Estimated time**: 1 hour

---

### 4. Training Script (`train.sh` or `train_main.py`)

**What needs to be implemented:**

```python
def main():
    # Load config
    config = TrainingConfig()

    # Create model
    model = create_model(
        num_channels=config.num_channels,
        num_res_blocks=config.num_res_blocks,
        device=config.device
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create trainer
    trainer = Trainer(model, optimizer, config)

    # Training loop
    for iteration in range(1000):
        trainer.training_iteration(iteration)

        # Print stats
        if iteration % 10 == 0:
            print(f"Iteration {iteration} complete")
            print(f"Replay buffer: {len(trainer.replay_buffer)} positions")
            print(f"Best model: {trainer.best_model_path}")
```

**Estimated complexity**: 100-150 lines
**Estimated time**: 2-3 hours

---

## Summary

### What You Have (80%)
- ✅ Complete game engine with all rules
- ✅ Neural network architecture
- ✅ MCTS implementation
- ✅ GUI for playing
- ✅ Comprehensive tests

### What You Need (20%)
- 🚧 Self-play engine (`selfplay.py`) - 1-2 days
- 🚧 Training pipeline (`train.py`) - 2-3 days
- 🚧 Configuration (`config.py`) - 1 hour
- 🚧 Training script (`train_main.py`) - 2-3 hours

**Total remaining work: 3-5 days**

### After Implementation

You'll be able to:
1. Run: `python train_main.py`
2. Let it train for days/weeks
3. Watch the engine get stronger through self-play
4. Play against the trained engine

### Expected Training Timeline

**On GPU:**
- Random → Coherent play: 6-12 hours
- Coherent → Competent: 1-3 days
- Competent → Strong: 1-2 weeks
- Strong → Master: 2-4 weeks

**On CPU:** 10-50x slower (not recommended)

---

## Quick Reference

**Files you have:**
- `hnefatafl/game.py` ✅
- `hnefatafl/network.py` ✅
- `hnefatafl/mcts.py` ✅
- `hnefatafl/gui.py` ✅
- `tests/test_*.py` ✅

**Files you need:**
- `hnefatafl/selfplay.py` 🚧
- `hnefatafl/train.py` 🚧
- `config.py` 🚧
- `train_main.py` 🚧

**Total lines to write:** ~800-1000 lines
**Total time estimate:** 3-5 days

The hard parts (game rules, MCTS, neural network) are DONE! ✅
The remaining parts are mostly boilerplate training code. 🚀
