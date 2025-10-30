"""
Training configuration for Copenhagen Hnefatafl AlphaZero.

This file contains all hyperparameters and settings for training.
You can modify these values to experiment with different training regimes.
"""

import torch
from dataclasses import dataclass


@dataclass
class MVPTrainingConfig:
    """
    Minimal Viable Product config for CPU testing.

    This is designed to complete in 30-60 minutes on a laptop CPU
    to verify the entire training pipeline works before spending
    money on GPU rentals.

    This will NOT produce a good model, but it will catch bugs!
    """

    # Self-play (minimal)
    num_simulations: int = 25  # Very few simulations
    num_games_per_iteration: int = 2  # Just 2 games
    temperature_threshold: int = 5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Training (tiny batches)
    replay_buffer_size: int = 1000
    batch_size: int = 16  # Small batches
    epochs_per_iteration: int = 2  # Few epochs
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_decay_steps: int = 10
    lr_decay_gamma: float = 0.95

    # Evaluation (minimal)
    num_eval_games: int = 4  # Just 4 games
    eval_win_threshold: float = 0.55
    eval_simulations: int = 25  # Very few simulations

    # Model (tiny for CPU)
    num_channels: int = 32  # Very small
    num_res_blocks: int = 2  # Very few blocks

    # Hardware
    device: str = 'cpu'
    num_workers: int = 1

    # Checkpointing
    checkpoint_dir: str = "checkpoints_mvp"
    save_interval: int = 2

    # Logging
    log_interval: int = 1
    verbose: bool = True


@dataclass
class GPUTestConfig:
    """
    Small GPU test config - verify GPU setup works.

    This runs 5 iterations with 5 games each.
    Games can go up to 1000 moves, with attackers winning on timeout
    (simulates the advantage of numerical superiority in long games).

    Time: ~30-60 minutes on GPU
    Cost: ~$0.20-0.50
    Purpose: Verify GPU works before committing to full training
    """

    # Self-play
    num_simulations: int = 800  # Standard MCTS depth
    num_games_per_iteration: int = 5  # 5 games per iteration
    temperature_threshold: int = 15
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_game_moves: int = 1000  # Long games allowed
    attacker_timeout_win: bool = True  # Attackers win on timeout

    # Training
    replay_buffer_size: int = 50000
    batch_size: int = 256  # Good GPU batch size
    epochs_per_iteration: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_decay_steps: int = 100
    lr_decay_gamma: float = 0.95

    # Evaluation
    num_eval_games: int = 20  # Decent evaluation
    eval_win_threshold: float = 0.55
    eval_simulations: int = 400

    # Model - standard size
    num_channels: int = 128
    num_res_blocks: int = 10

    # Hardware
    device: str = 'cpu'  # RTX 5090 needs PyTorch 2.7+ for CUDA sm_120 support
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints_gpu_test"
    save_interval: int = 2  # Save every 2 iterations

    # Logging
    log_interval: int = 1
    verbose: bool = True


@dataclass
class QuickTrainingConfig:
    """
    Quick training config for testing (CPU-friendly).

    Use this to verify the training pipeline works before
    committing to long GPU training runs.
    """

    # Self-play
    num_simulations: int = 100
    num_games_per_iteration: int = 10
    temperature_threshold: int = 10
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Training
    replay_buffer_size: int = 10000
    batch_size: int = 64
    epochs_per_iteration: int = 3
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_decay_steps: int = 20
    lr_decay_gamma: float = 0.95

    # Evaluation
    num_eval_games: int = 10
    eval_win_threshold: float = 0.55
    eval_simulations: int = 100

    # Model (small for CPU)
    num_channels: int = 64
    num_res_blocks: int = 3

    # Hardware
    device: str = 'cpu'
    num_workers: int = 1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5

    # Logging
    log_interval: int = 1
    verbose: bool = True


@dataclass
class StandardTrainingConfig:
    """
    Standard training config for GPU training.

    This is a balanced configuration for training on a single GPU
    over several days to achieve competent play.
    """

    # Self-play
    num_simulations: int = 800
    num_games_per_iteration: int = 100
    temperature_threshold: int = 15
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Training
    replay_buffer_size: int = 500000
    batch_size: int = 256
    epochs_per_iteration: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_decay_steps: int = 100
    lr_decay_gamma: float = 0.95

    # Evaluation
    num_eval_games: int = 40
    eval_win_threshold: float = 0.55
    eval_simulations: int = 400

    # Model
    num_channels: int = 128
    num_res_blocks: int = 10

    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5

    # Logging
    log_interval: int = 1
    verbose: bool = True


@dataclass
class IntenseTrainingConfig:
    """
    Intense training config for high-end GPU training.

    Use this if you have a powerful GPU (e.g., RTX 4090, A100) and
    want to train for weeks to achieve master-level play.
    """

    # Self-play
    num_simulations: int = 1600
    num_games_per_iteration: int = 500
    temperature_threshold: int = 15
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Training
    replay_buffer_size: int = 1000000
    batch_size: int = 512
    epochs_per_iteration: int = 20
    learning_rate: float = 0.002
    weight_decay: float = 1e-4
    lr_decay_steps: int = 50
    lr_decay_gamma: float = 0.9

    # Evaluation
    num_eval_games: int = 100
    eval_win_threshold: float = 0.55
    eval_simulations: int = 800

    # Model (larger for better capacity)
    num_channels: int = 256
    num_res_blocks: int = 20

    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 8

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 2  # Save more frequently

    # Logging
    log_interval: int = 1
    verbose: bool = True


def get_config(config_name: str = "standard"):
    """
    Get training configuration by name.

    Args:
        config_name: One of 'mvp', 'gpu_test', 'quick', 'standard', 'intense'

    Returns:
        Training configuration object
    """
    configs = {
        'mvp': MVPTrainingConfig(),
        'gpu_test': GPUTestConfig(),
        'quick': QuickTrainingConfig(),
        'standard': StandardTrainingConfig(),
        'intense': IntenseTrainingConfig()
    }

    if config_name not in configs:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Choose from: {list(configs.keys())}"
        )

    return configs[config_name]


def print_config_comparison():
    """Print comparison of all configs"""
    print("=" * 90)
    print("TRAINING CONFIGURATION COMPARISON")
    print("=" * 90)

    configs = {
        'MVP (CPU Test)': MVPTrainingConfig(),
        'Quick (CPU)': QuickTrainingConfig(),
        'Standard (GPU)': StandardTrainingConfig(),
        'Intense (High-end GPU)': IntenseTrainingConfig()
    }

    # Print table header
    print(f"\n{'Parameter':<30} {'MVP':<12} {'Quick':<12} {'Standard':<15} {'Intense':<15}")
    print("-" * 90)

    # Self-play parameters
    print("\n--- SELF-PLAY ---")
    print(f"{'Simulations per move':<30} "
          f"{configs['MVP (CPU Test)'].num_simulations:<12} "
          f"{configs['Quick (CPU)'].num_simulations:<12} "
          f"{configs['Standard (GPU)'].num_simulations:<15} "
          f"{configs['Intense (High-end GPU)'].num_simulations:<15}")

    print(f"{'Games per iteration':<30} "
          f"{configs['MVP (CPU Test)'].num_games_per_iteration:<12} "
          f"{configs['Quick (CPU)'].num_games_per_iteration:<12} "
          f"{configs['Standard (GPU)'].num_games_per_iteration:<15} "
          f"{configs['Intense (High-end GPU)'].num_games_per_iteration:<15}")

    # Training parameters
    print("\n--- TRAINING ---")
    print(f"{'Replay buffer size':<30} "
          f"{configs['MVP (CPU Test)'].replay_buffer_size:<12,} "
          f"{configs['Quick (CPU)'].replay_buffer_size:<12,} "
          f"{configs['Standard (GPU)'].replay_buffer_size:<15,} "
          f"{configs['Intense (High-end GPU)'].replay_buffer_size:<15,}")

    print(f"{'Batch size':<30} "
          f"{configs['MVP (CPU Test)'].batch_size:<12} "
          f"{configs['Quick (CPU)'].batch_size:<12} "
          f"{configs['Standard (GPU)'].batch_size:<15} "
          f"{configs['Intense (High-end GPU)'].batch_size:<15}")

    print(f"{'Epochs per iteration':<30} "
          f"{configs['MVP (CPU Test)'].epochs_per_iteration:<12} "
          f"{configs['Quick (CPU)'].epochs_per_iteration:<12} "
          f"{configs['Standard (GPU)'].epochs_per_iteration:<15} "
          f"{configs['Intense (High-end GPU)'].epochs_per_iteration:<15}")

    # Model parameters
    print("\n--- MODEL ---")
    print(f"{'Channels':<30} "
          f"{configs['MVP (CPU Test)'].num_channels:<12} "
          f"{configs['Quick (CPU)'].num_channels:<12} "
          f"{configs['Standard (GPU)'].num_channels:<15} "
          f"{configs['Intense (High-end GPU)'].num_channels:<15}")

    print(f"{'Residual blocks':<30} "
          f"{configs['MVP (CPU Test)'].num_res_blocks:<12} "
          f"{configs['Quick (CPU)'].num_res_blocks:<12} "
          f"{configs['Standard (GPU)'].num_res_blocks:<15} "
          f"{configs['Intense (High-end GPU)'].num_res_blocks:<15}")

    # Rough parameter counts
    def estimate_params(channels, blocks):
        # Very rough estimate
        return channels * channels * 3 * 3 * 2 * blocks * 1000

    print(f"{'Est. parameters (M)':<30} "
          f"{estimate_params(configs['MVP (CPU Test)'].num_channels, configs['MVP (CPU Test)'].num_res_blocks) / 1e6:<12.1f} "
          f"{estimate_params(configs['Quick (CPU)'].num_channels, configs['Quick (CPU)'].num_res_blocks) / 1e6:<12.1f} "
          f"{estimate_params(configs['Standard (GPU)'].num_channels, configs['Standard (GPU)'].num_res_blocks) / 1e6:<15.1f} "
          f"{estimate_params(configs['Intense (High-end GPU)'].num_channels, configs['Intense (High-end GPU)'].num_res_blocks) / 1e6:<15.1f}")

    # Evaluation
    print("\n--- EVALUATION ---")
    print(f"{'Eval games':<30} "
          f"{configs['MVP (CPU Test)'].num_eval_games:<12} "
          f"{configs['Quick (CPU)'].num_eval_games:<12} "
          f"{configs['Standard (GPU)'].num_eval_games:<15} "
          f"{configs['Intense (High-end GPU)'].num_eval_games:<15}")

    print(f"{'Eval simulations':<30} "
          f"{configs['MVP (CPU Test)'].eval_simulations:<12} "
          f"{configs['Quick (CPU)'].eval_simulations:<12} "
          f"{configs['Standard (GPU)'].eval_simulations:<15} "
          f"{configs['Intense (High-end GPU)'].eval_simulations:<15}")

    # Estimated time per iteration
    print("\n--- ESTIMATED TIME PER ITERATION ---")
    print(f"{'MVP (CPU Test)':<30} ~5-10 minutes")
    print(f"{'Quick (CPU)':<30} ~30-60 minutes")
    print(f"{'Standard (GPU)':<30} ~5-10 minutes")
    print(f"{'Intense (High-end GPU)':<30} ~10-20 minutes")

    print("\n" + "=" * 90)
    print("\nRECOMMENDATIONS:")
    print("-" * 90)
    print("• MVP: Use FIRST to verify everything works (30-60 min total)")
    print("• Quick: Use for testing the pipeline on CPU (2-3 hours)")
    print("• Standard: Use for actual training on consumer GPU (RTX 3060+)")
    print("• Intense: Use for high-end GPU training (RTX 4090, A100, etc.)")
    print("\nExpected training time to strong play:")
    print("• Standard config: 1-2 weeks")
    print("• Intense config: 3-7 days")
    print("\n⚠️  IMPORTANT: Always run MVP config first to catch bugs before GPU rental!")
    print("=" * 90)


if __name__ == "__main__":
    print_config_comparison()
