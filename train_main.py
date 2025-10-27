#!/usr/bin/env python3
"""
Main training script for Copenhagen Hnefatafl AlphaZero.

Usage:
    python train_main.py                    # Use standard config
    python train_main.py --config quick     # Use quick config for testing
    python train_main.py --config intense   # Use intense config for high-end GPU
    python train_main.py --iterations 100   # Train for 100 iterations
    python train_main.py --resume checkpoint.pt  # Resume from checkpoint
"""

import argparse
import sys
import os
import torch

from hnefatafl.train import Trainer, TrainingConfig
from config import get_config, print_config_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Train Copenhagen Hnefatafl AlphaZero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test the pipeline with quick config (CPU-friendly)
  python train_main.py --config quick --iterations 5

  # Train with standard config for 100 iterations
  python train_main.py --config standard --iterations 100

  # Resume training from checkpoint
  python train_main.py --resume checkpoints/checkpoint_iter_50.pt

  # Use intense config on high-end GPU
  python train_main.py --config intense --iterations 500

  # Show config comparison
  python train_main.py --show-configs
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        choices=['mvp', 'quick', 'standard', 'intense'],
        default='standard',
        help='Training configuration preset (default: standard)'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of training iterations (default: 1000)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--show-configs',
        action='store_true',
        help='Show comparison of all configs and exit'
    )

    # Advanced options
    parser.add_argument(
        '--num-simulations',
        type=int,
        default=None,
        help='Override number of MCTS simulations per move'
    )

    parser.add_argument(
        '--num-games',
        type=int,
        default=None,
        help='Override number of games per iteration'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override learning rate'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Override device (cpu or cuda)'
    )

    args = parser.parse_args()

    # Show configs and exit
    if args.show_configs:
        print_config_comparison()
        return

    # Print header
    print_header()

    # Load config
    print(f"\nLoading '{args.config}' configuration...")
    config = get_config(args.config)

    # Apply overrides
    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations
        print(f"   Override: num_simulations = {args.num_simulations}")

    if args.num_games is not None:
        config.num_games_per_iteration = args.num_games
        print(f"   Override: num_games_per_iteration = {args.num_games}")

    if args.batch_size is not None:
        config.batch_size = args.batch_size
        print(f"   Override: batch_size = {args.batch_size}")

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
        print(f"   Override: learning_rate = {args.learning_rate}")

    if args.device is not None:
        config.device = args.device
        print(f"   Override: device = {args.device}")

    # Check CUDA availability
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA requested but not available. Falling back to CPU.")
        print("   Training will be MUCH slower on CPU (~10-50x).")
        print("   Consider using --config quick for CPU testing.")
        response = input("\nContinue with CPU? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
        config.device = 'cpu'

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Checkpoint not found: {args.resume}")
            return

        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Print training plan
    print_training_plan(args.iterations, config)

    # Confirm before starting
    if not confirm_training():
        print("Training cancelled.")
        return

    # Start training!
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    try:
        trainer.train(num_iterations=args.iterations)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print("\nBest model saved to: checkpoints/best_model.pt")
        print("\nTo play against the trained model:")
        print("  python play.py --mode gui --game-mode cpu")
        print("\nTo continue training:")
        print(f"  python train_main.py --resume checkpoints/checkpoint_iter_{trainer.iteration}.pt")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Checkpoint saved at iteration {trainer.iteration}")
        print("\nTo resume training:")
        print(f"  python train_main.py --resume checkpoints/checkpoint_iter_{trainer.iteration}.pt")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nCheckpoint may be available at: checkpoints/")


def print_header():
    """Print program header"""
    print("=" * 70)
    print(" " * 15 + "COPENHAGEN HNEFATAFL")
    print(" " * 18 + "AlphaZero Training")
    print("=" * 70)


def print_training_plan(iterations: int, config: TrainingConfig):
    """Print summary of training plan"""
    print("\n" + "=" * 70)
    print("TRAINING PLAN")
    print("=" * 70)

    print(f"\nIterations: {iterations}")
    print(f"Device: {config.device}")

    # Estimate totals
    total_games = iterations * config.num_games_per_iteration
    avg_moves_per_game = 40  # Rough estimate
    total_positions = total_games * avg_moves_per_game

    print(f"\nEstimated totals:")
    print(f"  Games: ~{total_games:,}")
    print(f"  Positions: ~{total_positions:,}")

    # Estimate time
    if config.device == 'cuda':
        minutes_per_iter = 7 if config.num_simulations <= 800 else 15
    else:
        minutes_per_iter = 60 if config.num_simulations <= 200 else 180

    total_minutes = minutes_per_iter * iterations

    print(f"\nEstimated time: {format_duration(total_minutes)}")

    print(f"\nKey parameters:")
    print(f"  MCTS simulations: {config.num_simulations}")
    print(f"  Games per iteration: {config.num_games_per_iteration}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Model size: {config.num_channels} channels, {config.num_res_blocks} blocks")

    print("\n" + "=" * 70)


def format_duration(minutes: float) -> str:
    """Format duration in human-readable form"""
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    elif minutes < 1440:
        hours = minutes / 60
        return f"{hours:.1f} hours"
    else:
        days = minutes / 1440
        return f"{days:.1f} days"


def confirm_training() -> bool:
    """Ask user to confirm training"""
    try:
        response = input("\nStart training? (y/n): ")
        return response.lower() == 'y'
    except (EOFError, KeyboardInterrupt):
        return False


if __name__ == "__main__":
    main()
