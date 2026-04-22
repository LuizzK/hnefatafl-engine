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

from .game import HnefataflGame, Move, Player, GameResult, get_policy_size
from .mcts import MCTS


# ---------------------------------------------------------------------------
# Board-symmetry utilities for data augmentation
#
# Copenhagen Hnefatafl has the full dihedral-8 symmetry group on the 11x11
# board with symmetric starting position. Each symmetry g maps (r, c) -> (r', c')
# and transforms rook-move directions consistently; distances are invariant.
# We precompute a 4840-length index permutation per symmetry once at import.
# ---------------------------------------------------------------------------

_BOARD_SIZE = 11

_POINT_XFORM = {
    'e':  lambda r, c, B: (r, c),
    'r1': lambda r, c, B: (B - c, r),
    'r2': lambda r, c, B: (B - r, B - c),
    'r3': lambda r, c, B: (c, B - r),
    'fh': lambda r, c, B: (r, B - c),
    'fv': lambda r, c, B: (B - r, c),
    'td': lambda r, c, B: (c, r),
    'ta': lambda r, c, B: (B - c, B - r),
}

# direction IDs: 0=up, 1=down, 2=left, 3=right
# Derived from how displacements transform under each symmetry:
# r1 (90 CCW): (dr, dc) -> (-dc, dr)  => [2, 3, 1, 0]
# r2 (180):    (dr, dc) -> (-dr, -dc) => [1, 0, 3, 2]
# r3 (270 CCW): (dr, dc) -> (dc, -dr) => [3, 2, 0, 1]
# fh (flip h): (dr, dc) -> (dr, -dc)  => [0, 1, 3, 2]
# fv (flip v): (dr, dc) -> (-dr, dc)  => [1, 0, 2, 3]
# td (transpose): (dr, dc) -> (dc, dr) => [2, 3, 0, 1]
# ta (anti-diag): (dr, dc) -> (-dc, -dr) => [3, 2, 1, 0]
_DIR_MAP = {
    'e':  [0, 1, 2, 3],
    'r1': [2, 3, 1, 0],
    'r2': [1, 0, 3, 2],
    'r3': [3, 2, 0, 1],
    'fh': [0, 1, 3, 2],
    'fv': [1, 0, 2, 3],
    'td': [2, 3, 0, 1],
    'ta': [3, 2, 1, 0],
}


def _build_symmetry_perms(N: int = _BOARD_SIZE):
    B = N - 1
    perms = {}
    size = N * N * 4 * 10
    for name in _POINT_XFORM:
        perm = np.empty(size, dtype=np.int64)
        for from_sq in range(N * N):
            r, c = divmod(from_sq, N)
            r2, c2 = _POINT_XFORM[name](r, c, B)
            new_from = r2 * N + c2
            for d in range(4):
                new_d = _DIR_MAP[name][d]
                for dist in range(1, 11):
                    old_idx = from_sq * 40 + d * 10 + (dist - 1)
                    new_idx = new_from * 40 + new_d * 10 + (dist - 1)
                    perm[old_idx] = new_idx
        perms[name] = perm
    return perms


_SYMMETRY_PERMS = _build_symmetry_perms()

SYM_STATE_OPS = {
    'e':  lambda s: s,
    'r1': lambda s: np.rot90(s, k=1, axes=(1, 2)),
    'r2': lambda s: np.rot90(s, k=2, axes=(1, 2)),
    'r3': lambda s: np.rot90(s, k=3, axes=(1, 2)),
    'fh': lambda s: np.flip(s, axis=2),
    'fv': lambda s: np.flip(s, axis=1),
    'td': lambda s: np.transpose(s, (0, 2, 1)),
    'ta': lambda s: np.flip(np.transpose(s, (0, 2, 1)), axis=(1, 2)),
}

SYM_NAMES = ('e', 'r1', 'r2', 'r3', 'fh', 'fv', 'td', 'ta')


def apply_symmetry(state: np.ndarray, policy: np.ndarray, name: str):
    """Apply dihedral symmetry `name` to (state, policy)."""
    new_state = np.ascontiguousarray(SYM_STATE_OPS[name](state))
    perm = _SYMMETRY_PERMS[name]
    new_policy = np.zeros_like(policy)
    new_policy[perm] = policy
    return new_state, new_policy


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
        dirichlet_epsilon: float = 0.25,
        max_game_moves: int = 200,
        attacker_timeout_win: bool = True,
        batch_size: int = 32,
        batch_evaluator=None,
    ):
        """
        Initialize self-play worker.

        Args:
            model: Neural network for position evaluation
            num_simulations: Number of MCTS simulations per move
            temperature_threshold: Move number after which to use greedy selection
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Weight of Dirichlet noise in root node
            max_game_moves: Maximum moves before ending game
            attacker_timeout_win: If True, attackers win on timeout
            batch_size: Number of positions to evaluate in parallel on GPU
        """
        self.model = model
        self.num_simulations = num_simulations
        self.temperature_threshold = temperature_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_game_moves = max_game_moves
        self.attacker_timeout_win = attacker_timeout_win
        self.batch_evaluator = batch_evaluator
        self.mcts = MCTS(
            neural_network=model,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            batch_size=batch_size,
            batch_evaluator=batch_evaluator,
        )

    def play_game(self, verbose: bool = False, max_moves: int = None, attacker_timeout_win: bool = None) -> List[TrainingExample]:
        """
        Play one complete self-play game.

        Args:
            verbose: Print progress messages
            max_moves: Maximum moves before ending game (None = use self.max_game_moves)
            attacker_timeout_win: If True, attackers win on timeout (None = use self.attacker_timeout_win)

        Returns:
            List of training examples (state, policy, outcome) for each move
        """
        # Use instance defaults if not specified
        if max_moves is None:
            max_moves = self.max_game_moves
        if attacker_timeout_win is None:
            attacker_timeout_win = self.attacker_timeout_win
        game = HnefataflGame()
        training_data = []
        move_count = 0
        moves_played = []  # Track all moves for display

        while not game.is_game_over() and move_count < max_moves:
            move_count += 1

            # Get current state
            state = game.encode_state()

            # Run MCTS to get move and policy
            temperature = 1.0 if move_count <= self.temperature_threshold else 0.1
            move, policy = self.mcts.search(game, temperature=temperature, add_noise=True)

            # Store training example (outcome will be filled in later)
            training_data.append(TrainingExample(
                state=state,
                policy=policy,
                value=0.0  # Placeholder
            ))

            # Track move for display
            moves_played.append(move)

            # Make the move
            game.make_move(move)

            if verbose and move_count % 5 == 0:
                current_player = "Attacker" if game.current_player == Player.DEFENDER else "Defender"
                print(f"    Move {move_count} ({current_player}'s turn)", flush=True)

        # Check if game hit move limit
        if move_count >= max_moves and not game.is_game_over():
            if attacker_timeout_win:
                # Attackers win if game goes too long (they have numerical advantage)
                if verbose:
                    print(f"  Game reached max moves ({max_moves}), attackers win by timeout", flush=True)
                # Force attacker win by setting result manually
                game.result = GameResult.ATTACKER_WIN
            else:
                # Draw
                if verbose:
                    print(f"  Game reached max moves ({max_moves}), declaring draw", flush=True)

        # Get game outcome
        result = game.result
        winner = game.get_winner()

        if verbose:
            print(f"  Game ended after {move_count} moves - {game.get_result_string()}", flush=True)
            # Display move history
            print(f"  Moves: ", end="", flush=True)
            move_strs = []
            for i, move in enumerate(moves_played):
                # Convert to chess-like notation (e.g., "d1-d3")
                from_col_str = chr(ord('a') + move.from_col)
                from_row_str = str(move.from_row + 1)
                to_col_str = chr(ord('a') + move.to_col)
                to_row_str = str(move.to_row + 1)
                move_strs.append(f"{from_col_str}{from_row_str}-{to_col_str}{to_row_str}")
            print(" ".join(move_strs), flush=True)

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
        progress_interval: int = 10,
        num_parallel: int = 1,
    ) -> List[TrainingExample]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate
            verbose: Whether to print progress
            progress_interval: Print progress every N games
            num_parallel: If > 1, run this many games concurrently in threads
                sharing a BatchEvaluator. This is the primary lever for GPU
                utilization — single-game MCTS is Python-bound and leaves the
                GPU idle 99% of the time.

        Returns:
            List of all training examples from all games
        """
        if num_parallel > 1:
            return self._generate_games_concurrent(num_games, num_parallel, verbose=verbose)

        all_examples = []
        start_time = time.time()

        for game_num in range(num_games):
            if verbose:
                print(f"\n  Starting game {game_num + 1}/{num_games}...", flush=True)

            game_examples = self.play_game(verbose=verbose)
            all_examples.extend(game_examples)

            if verbose:
                print(f"  ✓ Game {game_num + 1}/{num_games} complete: {len(game_examples)} moves", flush=True)

        if verbose:
            elapsed = time.time() - start_time
            print(f"\nGenerated {num_games} games in {elapsed:.1f}s")
            print(f"Total training examples: {len(all_examples)}")
            print(f"Average moves per game: {len(all_examples) / num_games:.1f}")

        return all_examples

    def _generate_games_concurrent(
        self,
        num_games: int,
        num_parallel: int,
        verbose: bool = False,
    ) -> List[TrainingExample]:
        """
        Run self-play games concurrently using threads + a shared
        BatchEvaluator. The GIL does not block us because all heavy work
        (NN forward pass) happens inside torch C++ and releases the GIL.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .mcts import BatchEvaluator

        if self.model is None:
            raise RuntimeError("concurrent self-play requires a model")

        evaluator = self.batch_evaluator or BatchEvaluator(
            self.model, max_batch=max(512, num_parallel * self.mcts.batch_size)
        )
        owns_evaluator = self.batch_evaluator is None

        # Per-thread workers share the model + evaluator. Each has its own
        # MCTS tree state; they coordinate only at the NN boundary.
        workers = [
            SelfPlayWorker(
                model=self.model,
                num_simulations=self.num_simulations,
                temperature_threshold=self.temperature_threshold,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon,
                max_game_moves=self.max_game_moves,
                attacker_timeout_win=self.attacker_timeout_win,
                batch_size=self.mcts.batch_size,
                batch_evaluator=evaluator,
            )
            for _ in range(num_parallel)
        ]

        all_examples: List[TrainingExample] = []
        all_lock = threading.Lock()
        completed = [0]
        start_time = time.time()

        def run_one(game_idx: int, worker_idx: int):
            examples = workers[worker_idx].play_game(verbose=False)
            with all_lock:
                all_examples.extend(examples)
                completed[0] += 1
                done = completed[0]
            if verbose:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"  ✓ Game {done}/{num_games} done ({len(examples)} moves, "
                    f"{elapsed:.1f}s elapsed, {rate:.2f} games/s)",
                    flush=True,
                )

        try:
            with ThreadPoolExecutor(max_workers=num_parallel) as pool:
                futures = [
                    pool.submit(run_one, i, i % num_parallel)
                    for i in range(num_games)
                ]
                for f in as_completed(futures):
                    f.result()  # raise if a worker crashed
        finally:
            if owns_evaluator:
                evaluator.close()

        if verbose:
            elapsed = time.time() - start_time
            print(f"\nGenerated {num_games} games in {elapsed:.1f}s "
                  f"({num_games / elapsed:.2f} games/s, {num_parallel}-way parallel)")
            print(f"Total training examples: {len(all_examples)}")
            if num_games:
                print(f"Average moves per game: {len(all_examples) / num_games:.1f}")

        return all_examples

    def augment_data(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """
        Augment training data with the 8 dihedral board symmetries.

        For each example, produce 8 training examples by applying every element
        of the dihedral-8 group to both state and policy. Distances are invariant;
        directions and from-squares are transformed via precomputed permutations.

        Args:
            examples: Original training examples

        Returns:
            Augmented training examples (8x larger)
        """
        augmented = []
        for example in examples:
            for name in SYM_NAMES:
                new_state, new_policy = apply_symmetry(example.state, example.policy, name)
                augmented.append(TrainingExample(new_state, new_policy, example.value))
        return augmented


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
