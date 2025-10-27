"""
Training pipeline for Copenhagen Hnefatafl AlphaZero.

This module implements the training loop:
1. Generate self-play games
2. Add to replay buffer
3. Train neural network
4. Evaluate new model vs old model
5. Replace if better
6. Save checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
import os
import random
from collections import deque

from .game import HnefataflGame, Player, GameResult
from .network import HnefataflNetwork, PolicyValueLoss
from .mcts import MCTS
from .selfplay import SelfPlayWorker, TrainingExample, ParallelSelfPlay


@dataclass
class TrainingConfig:
    """Configuration for training"""

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
    lr_decay_steps: int = 100  # Decay learning rate every N iterations
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
    save_interval: int = 5  # Save every N iterations

    # Logging
    log_interval: int = 1
    verbose: bool = True


class Trainer:
    """AlphaZero-style training pipeline"""

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Create model
        self.model = HnefataflNetwork(
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
            board_size=11
        ).to(self.device)

        # Keep a copy of the best model for evaluation
        self.best_model = HnefataflNetwork(
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
            board_size=11
        ).to(self.device)
        self.best_model.load_state_dict(self.model.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_steps,
            gamma=config.lr_decay_gamma
        )

        # Replay buffer
        self.replay_buffer: deque[TrainingExample] = deque(
            maxlen=config.replay_buffer_size
        )

        # Training statistics
        self.iteration = 0
        self.total_games = 0
        self.total_positions = 0
        self.win_rates = []
        self.losses = []

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        if config.verbose:
            print("=" * 70)
            print("COPENHAGEN HNEFATAFL - AlphaZero Training")
            print("=" * 70)
            print(f"Device: {self.device}")
            print(f"Model parameters: {self._count_parameters():,}")
            print(f"Replay buffer size: {config.replay_buffer_size:,}")
            print(f"Batch size: {config.batch_size}")
            print(f"Learning rate: {config.learning_rate}")
            print("=" * 70)

    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def training_iteration(self):
        """
        Execute one full training iteration:
        1. Generate self-play games
        2. Add to replay buffer
        3. Train network
        4. Evaluate new model
        5. Update best model if better
        6. Save checkpoint
        """
        self.iteration += 1
        start_time = time.time()

        print(f"\n{'='*70}", flush=True)
        print(f"ITERATION {self.iteration}", flush=True)
        print(f"{'='*70}", flush=True)

        # Step 1: Generate self-play games
        print("\n[1/5] Generating self-play games...", flush=True)
        new_examples = self._generate_selfplay_games()
        self.replay_buffer.extend(new_examples)
        self.total_games += self.config.num_games_per_iteration
        self.total_positions += len(new_examples)

        print(f"   Generated {len(new_examples)} new positions", flush=True)
        print(f"   Replay buffer: {len(self.replay_buffer):,} positions", flush=True)

        # Step 2: Train network
        print("\n[2/5] Training neural network...", flush=True)
        train_metrics = self._train_network()

        # Step 3: Evaluate new model
        print("\n[3/5] Evaluating new model...", flush=True)
        win_rate = self._evaluate_models()
        self.win_rates.append(win_rate)

        # Step 4: Update best model
        print("\n[4/5] Updating best model...", flush=True)
        if win_rate > self.config.eval_win_threshold:
            print(f"   ✓ New model wins {win_rate:.1%}! Replacing best model.", flush=True)
            self.best_model.load_state_dict(self.model.state_dict())
        else:
            print(f"   ✗ New model only wins {win_rate:.1%}. Keeping old model.", flush=True)
            # Revert to best model
            self.model.load_state_dict(self.best_model.state_dict())

        # Step 5: Save checkpoint
        if self.iteration % self.config.save_interval == 0:
            print("\n[5/5] Saving checkpoint...", flush=True)
            self._save_checkpoint()

        # Update learning rate
        self.scheduler.step()

        # Print iteration summary
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"ITERATION {self.iteration} COMPLETE ({elapsed:.1f}s)")
        print(f"{'='*70}")
        print(f"Total games: {self.total_games}")
        print(f"Total positions: {self.total_positions:,}")
        print(f"Win rate: {win_rate:.1%}")
        print(f"Policy loss: {train_metrics['policy_loss']:.4f}")
        print(f"Value loss: {train_metrics['value_loss']:.4f}")
        print(f"Total loss: {train_metrics['total_loss']:.4f}")
        print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*70}\n")

    def _generate_selfplay_games(self) -> List[TrainingExample]:
        """Generate self-play games using best model"""
        worker = SelfPlayWorker(
            model=self.best_model,
            num_simulations=self.config.num_simulations,
            temperature_threshold=self.config.temperature_threshold,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon
        )

        examples = worker.generate_games(
            num_games=self.config.num_games_per_iteration,
            verbose=self.config.verbose,
            progress_interval=1  # Show progress every game (was 10)
        )

        return examples

    def _train_network(self) -> Dict[str, float]:
        """
        Train neural network on replay buffer.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        loss_fn = PolicyValueLoss()

        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        num_batches = 0

        # Convert buffer to list for sampling
        buffer_list = list(self.replay_buffer)

        for epoch in range(self.config.epochs_per_iteration):
            # Shuffle data
            random.shuffle(buffer_list)

            # Create batches
            for i in range(0, len(buffer_list), self.config.batch_size):
                batch = buffer_list[i:i + self.config.batch_size]

                # Prepare batch tensors
                states = torch.stack([
                    torch.FloatTensor(ex.state) for ex in batch
                ]).to(self.device)

                target_policies = torch.stack([
                    torch.FloatTensor(ex.policy) for ex in batch
                ]).to(self.device)

                target_values = torch.FloatTensor([
                    ex.value for ex in batch
                ]).unsqueeze(1).to(self.device)

                # Forward pass
                policy_logits, value_pred = self.model(states)

                # Compute loss
                batch_total_loss, batch_policy_loss, batch_value_loss = loss_fn(
                    policy_logits, value_pred,
                    target_policies, target_values
                )

                # Backward pass
                self.optimizer.zero_grad()
                batch_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Accumulate losses
                total_loss += batch_total_loss.item()
                policy_loss += batch_policy_loss.item()
                value_loss += batch_value_loss.item()
                num_batches += 1

        # Average losses
        metrics = {
            'total_loss': total_loss / num_batches,
            'policy_loss': policy_loss / num_batches,
            'value_loss': value_loss / num_batches
        }

        self.losses.append(metrics)

        if self.config.verbose:
            print(f"   Trained on {len(buffer_list):,} positions", flush=True)
            print(f"   Total loss: {metrics['total_loss']:.4f}", flush=True)
            print(f"   Policy loss: {metrics['policy_loss']:.4f}", flush=True)
            print(f"   Value loss: {metrics['value_loss']:.4f}", flush=True)

        return metrics

    def _evaluate_models(self) -> float:
        """
        Evaluate new model against best model.

        Returns:
            Win rate of new model (0.0 to 1.0)
        """
        self.model.eval()
        self.best_model.eval()

        new_mcts = MCTS(
            neural_network=self.model,
            num_simulations=self.config.eval_simulations
        )
        old_mcts = MCTS(
            neural_network=self.best_model,
            num_simulations=self.config.eval_simulations
        )

        wins = 0
        losses = 0
        draws = 0

        for game_num in range(self.config.num_eval_games):
            # Alternate which model plays as attacker
            if game_num % 2 == 0:
                # New model is attacker
                result = self._play_evaluation_game(new_mcts, old_mcts)
                if result == Player.ATTACKER:
                    wins += 1
                elif result == Player.DEFENDER:
                    losses += 1
                else:
                    draws += 1
            else:
                # Old model is attacker
                result = self._play_evaluation_game(old_mcts, new_mcts)
                if result == Player.DEFENDER:
                    wins += 1
                elif result == Player.ATTACKER:
                    losses += 1
                else:
                    draws += 1

            if self.config.verbose and (game_num + 1) % 10 == 0:
                current_win_rate = wins / (game_num + 1)
                print(f"   Evaluated {game_num + 1}/{self.config.num_eval_games} games "
                      f"(win rate: {current_win_rate:.1%})", flush=True)

        win_rate = wins / self.config.num_eval_games

        if self.config.verbose:
            print(f"   Final: {wins}W {losses}L {draws}D (win rate: {win_rate:.1%})")

        return win_rate

    def _play_evaluation_game(
        self,
        attacker_mcts: MCTS,
        defender_mcts: MCTS
    ) -> Optional[Player]:
        """
        Play one evaluation game between two MCTS instances.

        Args:
            attacker_mcts: MCTS for attacker
            defender_mcts: MCTS for defender

        Returns:
            Winner (Player.ATTACKER, Player.DEFENDER, or None for draw)
        """
        game = HnefataflGame()
        move_count = 0
        max_moves = 200  # Prevent infinite games

        while not game.is_game_over() and move_count < max_moves:
            if game.current_player == Player.ATTACKER:
                move, _ = attacker_mcts.search(game, temperature=0.0)
            else:
                move, _ = defender_mcts.search(game, temperature=0.0)

            game.make_move(move)
            move_count += 1

        return game.get_winner()

    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_iter_{self.iteration}.pt"
        )

        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'best_model_state_dict': self.best_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'total_games': self.total_games,
            'total_positions': self.total_positions,
            'win_rates': self.win_rates,
            'losses': self.losses
        }

        torch.save(checkpoint, checkpoint_path)

        # Also save best model separately
        best_path = os.path.join(
            self.config.checkpoint_dir,
            "best_model.pt"
        )
        torch.save({
            'model_state_dict': self.best_model.state_dict(),
            'iteration': self.iteration,
            'config': self.config
        }, best_path)

        if self.config.verbose:
            print(f"   Saved checkpoint: {checkpoint_path}")
            print(f"   Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model.load_state_dict(checkpoint['best_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.total_games = checkpoint['total_games']
        self.total_positions = checkpoint['total_positions']
        self.win_rates = checkpoint['win_rates']
        self.losses = checkpoint['losses']

        print(f"Loaded checkpoint from iteration {self.iteration}")

    def train(self, num_iterations: int):
        """
        Run full training loop.

        Args:
            num_iterations: Number of training iterations to run
        """
        print(f"\nStarting training for {num_iterations} iterations...")
        print(f"Estimated time: {self._estimate_time(num_iterations)}")

        start_time = time.time()

        for _ in range(num_iterations):
            self.training_iteration()

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed / 3600:.1f} hours")
        print(f"Total games: {self.total_games}")
        print(f"Total positions: {self.total_positions:,}")
        print(f"Final win rate: {self.win_rates[-1]:.1%}")
        print(f"{'='*70}\n")

    def _estimate_time(self, num_iterations: int) -> str:
        """Estimate training time"""
        # Very rough estimate: 5-10 minutes per iteration on GPU
        minutes_per_iter = 7 if self.device.type == 'cuda' else 70
        total_minutes = minutes_per_iter * num_iterations

        if total_minutes < 60:
            return f"~{total_minutes:.0f} minutes"
        elif total_minutes < 1440:
            return f"~{total_minutes / 60:.1f} hours"
        else:
            return f"~{total_minutes / 1440:.1f} days"


def test_training():
    """Test training pipeline with small config"""
    print("Testing training pipeline...")
    print("=" * 70)

    # Create small config for testing
    config = TrainingConfig(
        num_simulations=50,
        num_games_per_iteration=2,
        batch_size=32,
        epochs_per_iteration=2,
        num_eval_games=4,
        num_channels=32,
        num_res_blocks=2,
        replay_buffer_size=1000,
        device='cpu',
        verbose=True
    )

    # Create trainer
    trainer = Trainer(config)

    # Run one iteration
    print("\nRunning one training iteration...")
    print("(This will take several minutes...)")
    trainer.training_iteration()

    print("\n" + "=" * 70)
    print("Training pipeline test PASSED! ✓")


if __name__ == "__main__":
    test_training()
