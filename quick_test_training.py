"""
Quick training test with 5-move games to verify pipeline works.
Should complete in ~2 minutes.
"""

import sys
from hnefatafl.train import Trainer, TrainingConfig

print("=" * 70)
print("QUICK TRAINING TEST (5-move games)")
print("=" * 70)
print("\nThis test will:")
print("  - Run 1 training iteration")
print("  - Play 2 games with max 5 moves each (forced draws)")
print("  - Train the network on those positions")
print("  - Evaluate the model (2 games)")
print("\nExpected time: ~2 minutes")
print("\nStarting in 3 seconds...")
print("=" * 70)

import time
time.sleep(3)

# Create test config with very short games
config = TrainingConfig(
    # Self-play - ultra minimal
    num_simulations=10,  # Very few simulations
    num_games_per_iteration=2,  # Just 2 games
    temperature_threshold=2,

    # Training
    batch_size=8,  # Small batch
    epochs_per_iteration=2,  # Few epochs

    # Evaluation - minimal
    num_eval_games=2,  # Just 2 eval games
    eval_simulations=10,

    # Model - tiny
    num_channels=16,  # Very small
    num_res_blocks=1,  # Just 1 block

    # Buffer
    replay_buffer_size=100,

    # Hardware
    device='cpu',

    # Checkpointing
    checkpoint_dir='checkpoints_test',
    save_interval=1,

    # Logging
    verbose=True
)

# Create trainer
print("\nInitializing trainer...")
trainer = Trainer(config)

# Monkey-patch to force 5-move games
original_play_game = trainer.best_model.__class__.__module__

print("\nModifying self-play to use 5-move max...")

# Import the SelfPlayWorker and patch it
from hnefatafl import selfplay
original_play_game_method = selfplay.SelfPlayWorker.play_game

def short_play_game(self, verbose=False, max_moves=5):
    """Play game with forced 5-move limit"""
    return original_play_game_method(self, verbose=verbose, max_moves=5)

selfplay.SelfPlayWorker.play_game = short_play_game

print("✓ Games will end after 5 moves (forced draws)\n")

# Run ONE iteration
print("=" * 70)
print("RUNNING 1 TEST ITERATION")
print("=" * 70)

try:
    trainer.training_iteration()

    print("\n" + "=" * 70)
    print("✅ TEST PASSED! Pipeline works correctly!")
    print("=" * 70)
    print("\nWhat worked:")
    print("  ✓ Self-play generated games")
    print("  ✓ Positions stored in replay buffer")
    print("  ✓ Neural network trained successfully")
    print("  ✓ Model evaluation completed")
    print("  ✓ Checkpoint saved")
    print("\nYou're safe to run the full MVP training!")
    print("Run: ./venv/bin/python train_main.py --config mvp --iterations 5 --yes")
    print("=" * 70)

except Exception as e:
    print("\n" + "=" * 70)
    print("❌ TEST FAILED!")
    print("=" * 70)
    print(f"\nError: {e}")
    print("\nPlease report this error.")
    import traceback
    traceback.print_exc()
    sys.exit(1)
