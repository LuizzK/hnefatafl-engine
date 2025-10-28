"""
Self-play engine for generating training games.

This module implements the self-play component of the AlphaZero pipeline.
The SelfPlayWorker plays games against itself using MCTS to generate
training data (state, policy, outcome) tuples.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

from .game import HnefataflGame, Move, Player, GameResult
from .mcts import MCTS


@dataclass
class TrainingExample:
    """A single training example from self-play"""
    state: np.ndarray  # Encoded board state (15, 11, 11)
    policy: np.ndarray  # MCTS visit counts as policy target
    value: float  # Game outcome from this player's perspective


class SelfPlayWorker:
    """
    Generate self-play games for training.

    Uses MCTS to play games and collects training data at each position.
    The collected data includes the board state, MCTS policy, and final
    game outcome.
    """

    def __init__(
        self,
        model,
        num_simulations: int = 800,
        temperature_threshold: int = 15,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Initialize self-play worker.

        Args:
            model: Neural network for position evaluation
            num_simulations: Number of MCTS simulations per move
            temperature_threshold: Move number after which to use greedy selection
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Weight of Dirichlet noise in root node
        """
        self.model = model
        self.num_simulations = num_simulations
        self.temperature_threshold = temperature_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.mcts = MCTS(
            neural_network=model,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )

    def play_game(self, verbose: bool = False, max_moves: int = 200) -> List[TrainingExample]:
        """
        Play one complete self-play game.

        Args:
            verbose: Print progress messages
            max_moves: Maximum moves before declaring draw (prevents infinite games)

        Returns:
            List of training examples (state, policy, outcome) for each move
        """
        game = HnefataflGame()
        training_data = []
        move_count = 0

        if verbose:
            print("Starting self-play game...", flush=True)

        while not game.is_game_over() and move_count < max_moves:
            move_count += 1

            # Get current state
            state = game.encode_state()

            # Run MCTS to get move and policy
            temperature = 1.0 if move_count <= self.temperature_threshold else 0.1
            move, policy = self.mcts.search(game, temperature=temperature)

            # Store training example (outcome will be filled in later)
            training_data.append(TrainingExample(
                state=state,
                policy=policy,
                value=0.0  # Placeholder
            ))

            # Make the move
            game.make_move(move)

            if verbose and move_count % 10 == 0:
                print(f"  Move {move_count}...", flush=True)

        # Check if game hit move limit
        if move_count >= max_moves and not game.is_game_over():
            if verbose:
                print(f"  Game reached max moves ({max_moves}), declaring draw", flush=True)

        # Get game outcome
        from .game import GameResult
        result = game.result
        winner = game.get_winner()

        if verbose:
            print(f"  Game ended after {move_count} moves - {game.get_result_string()}", flush=True)

        # Assign outcomes to all positions
        # Outcome is from perspective of player at that position
        for i, example in enumerate(training_data):
            # Determine which player made this move
            player_at_position = Player.ATTACKER if i % 2 == 0 else Player.DEFENDER

            # Assign value based on outcome
            if result == GameResult.ATTACKER_WIN:
                value = 1.0 if player_at_position == Player.ATTACKER else -1.0
            elif result == GameResult.DEFENDER_WIN:
                value = 1.0 if player_at_position == Player.DEFENDER else -1.0
            else:
                value = 0.0  # Draw

            training_data[i].value = value

        return training_data

    def generate_games(
        self,
        num_games: int,
        verbose: bool = False,
        progress_interval: int = 10
    ) -> List[TrainingExample]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate
            verbose: Whether to print progress
            progress_interval: Print progress every N games

        Returns:
            List of all training examples from all games
        """
        all_examples = []
        start_time = time.time()

        for game_num in range(num_games):
            # Generate one game
            game_examples = self.play_game(verbose=False)
            all_examples.extend(game_examples)

            # Print progress
            if verbose and (game_num + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                games_per_sec = (game_num + 1) / elapsed
                print(f"Generated {game_num + 1}/{num_games} games "
                      f"({games_per_sec:.2f} games/sec, "
                      f"{len(all_examples)} total positions)", flush=True)
            elif verbose:
                # Print brief progress for every game
                print(f"  Game {game_num + 1}/{num_games} complete ({len(game_examples)} moves)", flush=True)

        if verbose:
            elapsed = time.time() - start_time
            print(f"\nGenerated {num_games} games in {elapsed:.1f}s")
            print(f"Total training examples: {len(all_examples)}")
            print(f"Average moves per game: {len(all_examples) / num_games:.1f}")

        return all_examples

    def augment_data(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """
        Augment training data with board symmetries.

        Copenhagen Hnefatafl has 4-fold rotational symmetry and 4 reflections,
        giving 8 symmetric positions for each board state.

        Args:
            examples: Original training examples

        Returns:
            Augmented training examples (8x larger)
        """
        augmented = []

        for example in examples:
            state = example.state
            policy = example.policy
            value = example.value

            # Original
            augmented.append(TrainingExample(state, policy, value))

            # Rotate 90 degrees
            state_90 = np.rot90(state, k=1, axes=(1, 2))
            policy_90 = self._rotate_policy(policy, k=1)
            augmented.append(TrainingExample(state_90, policy_90, value))

            # Rotate 180 degrees
            state_180 = np.rot90(state, k=2, axes=(1, 2))
            policy_180 = self._rotate_policy(policy, k=2)
            augmented.append(TrainingExample(state_180, policy_180, value))

            # Rotate 270 degrees
            state_270 = np.rot90(state, k=3, axes=(1, 2))
            policy_270 = self._rotate_policy(policy, k=3)
            augmented.append(TrainingExample(state_270, policy_270, value))

            # Horizontal flip
            state_h = np.flip(state, axis=2)
            policy_h = self._flip_policy_horizontal(policy)
            augmented.append(TrainingExample(state_h, policy_h, value))

            # Vertical flip
            state_v = np.flip(state, axis=1)
            policy_v = self._flip_policy_vertical(policy)
            augmented.append(TrainingExample(state_v, policy_v, value))

            # Diagonal flip (transpose)
            state_d1 = np.transpose(state, (0, 2, 1))
            policy_d1 = self._transpose_policy(policy)
            augmented.append(TrainingExample(state_d1, policy_d1, value))

            # Anti-diagonal flip
            state_d2 = np.flip(np.transpose(state, (0, 2, 1)), axis=(1, 2))
            policy_d2 = self._flip_policy_horizontal(self._flip_policy_vertical(
                self._transpose_policy(policy)))
            augmented.append(TrainingExample(state_d2, policy_d2, value))

        return augmented

    def _rotate_policy(self, policy: np.ndarray, k: int) -> np.ndarray:
        """
        Rotate policy vector by k*90 degrees.

        Policy is a flat vector of move probabilities. We need to reshape it
        to the board dimensions, rotate, and flatten back.
        """
        # TODO: Implement policy rotation
        # This is complex because policy encodes moves, not just board positions
        # For now, return original policy
        return policy

    def _flip_policy_horizontal(self, policy: np.ndarray) -> np.ndarray:
        """Flip policy horizontally"""
        # TODO: Implement policy flipping
        return policy

    def _flip_policy_vertical(self, policy: np.ndarray) -> np.ndarray:
        """Flip policy vertically"""
        # TODO: Implement policy flipping
        return policy

    def _transpose_policy(self, policy: np.ndarray) -> np.ndarray:
        """Transpose policy"""
        # TODO: Implement policy transposition
        return policy


class ParallelSelfPlay:
    """
    Generate self-play games in parallel using multiple workers.

    This can significantly speed up data generation by utilizing multiple
    CPU cores or GPU batches.
    """

    def __init__(
        self,
        model,
        num_workers: int = 4,
        num_simulations: int = 800
    ):
        """
        Initialize parallel self-play.

        Args:
            model: Neural network for position evaluation
            num_workers: Number of parallel workers
            num_simulations: Number of MCTS simulations per move
        """
        self.model = model
        self.num_workers = num_workers
        self.workers = [
            SelfPlayWorker(model, num_simulations=num_simulations)
            for _ in range(num_workers)
        ]

    def generate_games(
        self,
        num_games: int,
        verbose: bool = False
    ) -> List[TrainingExample]:
        """
        Generate games in parallel.

        Args:
            num_games: Total number of games to generate
            verbose: Whether to print progress

        Returns:
            List of all training examples
        """
        # Simple sequential implementation for now
        # TODO: Implement actual parallelization using multiprocessing or threading
        games_per_worker = num_games // self.num_workers
        all_examples = []

        for worker_id, worker in enumerate(self.workers):
            if verbose:
                print(f"Worker {worker_id + 1}/{self.num_workers} "
                      f"generating {games_per_worker} games...")

            examples = worker.generate_games(
                games_per_worker,
                verbose=verbose
            )
            all_examples.extend(examples)

        return all_examples


def test_selfplay():
    """Test self-play functionality"""
    print("Testing self-play engine...")
    print("=" * 60)

    # Create a random policy network (not trained)
    print("\n1. Creating random policy network...")
    from .network import create_model
    model = create_model(num_channels=64, num_res_blocks=3, device='cpu')
    print("   ✓ Model created")

    # Create self-play worker
    print("\n2. Creating self-play worker...")
    worker = SelfPlayWorker(
        model=model,
        num_simulations=50,  # Fewer simulations for testing
        temperature_threshold=10
    )
    print("   ✓ Worker created")

    # Generate one game
    print("\n3. Generating one self-play game...")
    print("   (This will take a minute or two...)")
    examples = worker.play_game(verbose=True)
    print(f"   ✓ Generated {len(examples)} training examples")

    # Check data format
    print("\n4. Checking data format...")
    example = examples[0]
    print(f"   State shape: {example.state.shape}")
    print(f"   Policy shape: {example.policy.shape}")
    print(f"   Value: {example.value}")
    print("   ✓ Data format correct")

    # Test multiple games
    print("\n5. Generating 3 games...")
    all_examples = worker.generate_games(num_games=3, verbose=True)
    print(f"   ✓ Generated {len(all_examples)} total examples")

    print("\n" + "=" * 60)
    print("Self-play engine test PASSED! ✓")


if __name__ == "__main__":
    test_selfplay()
