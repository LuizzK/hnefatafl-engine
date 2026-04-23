"""
Monte Carlo Tree Search (MCTS) implementation for Copenhagen Hnefatafl

Uses PUCT (Predictor + Upper Confidence bounds applied to Trees) algorithm
with neural network guidance, similar to AlphaZero.
"""

import numpy as np
import math
import threading
import time
from typing import List, Dict, Optional, Tuple
from hnefatafl.game import HnefataflGame, Move, Player


class BatchEvaluator:
    """
    Thread-safe NN evaluation batcher for concurrent self-play.

    Multiple self-play threads call `evaluate(states)` and block until the
    shared worker thread drains the queue and runs one batched GPU call.
    Without this, each MCTS runs its own tiny NN call; the GPU sits idle
    99% of the time because the CPU-side tree walk dominates.
    """

    def __init__(self, model, max_batch: int = 512, max_wait_ms: float = 2.0):
        import torch
        self._torch = torch
        self._F = __import__('torch.nn.functional', fromlist=['F'])
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: list = []  # (states_np, event, box)
        self._alive = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def evaluate(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Submit a (N, 15, 11, 11) batch. Blocks until results are ready."""
        event = threading.Event()
        box: dict = {}
        with self._cond:
            self._pending.append((states, event, box))
            self._cond.notify()
        event.wait()
        return box['policies'], box['values']

    def _drain(self):
        with self._cond:
            # Wait briefly for more requests to accumulate.
            if not self._pending:
                self._cond.wait(timeout=self.max_wait)
            reqs = self._pending
            self._pending = []
        return reqs

    def _run(self):
        torch = self._torch
        F = self._F
        while self._alive:
            reqs = self._drain()
            if not reqs:
                continue
            try:
                all_states = np.concatenate([r[0] for r in reqs], axis=0)
                state_tensor = torch.from_numpy(all_states).float().to(
                    self.device, non_blocking=True
                )
                with torch.no_grad():
                    logits, values = self.model.forward(state_tensor)
                    probs = F.softmax(logits, dim=1)
                policies_np = probs.detach().cpu().numpy()
                values_np = values.detach().cpu().numpy().flatten()
            except Exception as exc:  # propagate to all waiters so they don't hang
                for _, event, box in reqs:
                    box['error'] = exc
                    event.set()
                continue
            offset = 0
            for states, event, box in reqs:
                n = len(states)
                box['policies'] = policies_np[offset:offset + n]
                box['values'] = values_np[offset:offset + n]
                offset += n
                event.set()

    def close(self):
        self._alive = False
        with self._cond:
            self._cond.notify_all()


class MCTSNode:
    """
    Node in the MCTS tree.

    Each node represents a game state and stores statistics about
    visits, values, and prior probabilities for actions.
    """

    def __init__(self, game_state: HnefataflGame, parent: Optional['MCTSNode'] = None,
                 prior_prob: float = 0.0, move: Optional[Move] = None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior_prob = prior_prob  # Prior probability from neural network

        self.children: Dict[Move, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0  # For batched MCTS - prevents parallel searches from visiting same node

        self.is_expanded = False

    def get_value(self) -> float:
        """Get average value (Q-value) of this node, accounting for virtual loss"""
        if self.visit_count == 0:
            return 0.0
        # Virtual loss reduces Q-value to discourage parallel searches from picking same node
        return (self.value_sum - self.virtual_loss) / self.visit_count

    def get_ucb_score(self, c_puct: float = 1.5, parent_visits: int = 1) -> float:
        """
        Calculate UCB score using PUCT formula.

        UCB = Q + c_puct * P * sqrt(parent_N) / (1 + N)

        where:
        - Q is the average value
        - P is the prior probability
        - N is the visit count
        - c_puct is the exploration constant
        """
        q_value = self.get_value()

        # Exploration term
        u_value = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count)

        return q_value + u_value

    def select_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_child = None

        for child in self.children.values():
            score = child.get_ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy_probs: np.ndarray, legal_moves: List[Move]):
        """
        Expand node by creating children for all legal moves.

        Priors are drawn from the neural network policy vector indexed by
        encode_move(move), clipped to non-negative, and renormalized over
        legal moves. Falls back to uniform if the legal mass is ~0
        (untrained network edge case).

        Args:
            policy_probs: Policy probabilities from neural network (size get_policy_size())
            legal_moves: List of legal moves from this position
        """
        from hnefatafl.game import encode_move, get_policy_size

        if self.is_expanded:
            return

        assert policy_probs.shape == (get_policy_size(),), \
            f"expected policy size {get_policy_size()}, got {policy_probs.shape}"

        legal_indices = np.array([encode_move(m) for m in legal_moves], dtype=np.int64)
        legal_priors = np.maximum(policy_probs[legal_indices].astype(np.float64), 0.0)
        total = legal_priors.sum()
        if total > 1e-8:
            legal_priors /= total
        else:
            legal_priors = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float64)

        for move, prior in zip(legal_moves, legal_priors):
            child_game = self.game_state.copy()
            # Move came from get_legal_moves()— skip redundant validation.
            child_game.make_move(move, _assume_legal=True)
            child_node = MCTSNode(child_game, parent=self, prior_prob=float(prior), move=move)
            self.children[move] = child_node

        self.is_expanded = True

    def update(self, value: float):
        """
        Update node statistics after a simulation.

        Args:
            value: Value to backpropagate (from perspective of player who just moved)
        """
        self.visit_count += 1
        self.value_sum += value

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not expanded)"""
        return not self.is_expanded


class MCTS:
    """
    Monte Carlo Tree Search engine for Copenhagen Hnefatafl.

    Uses neural network to guide the search and evaluate positions.
    """

    def __init__(self, neural_network=None, num_simulations: int = 800,
                 c_puct: float = 1.5, temperature: float = 1.0,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 batch_size: int = 32,
                 batch_evaluator: Optional['BatchEvaluator'] = None):
        """
        Initialize MCTS.

        Args:
            neural_network: Neural network for position evaluation
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for PUCT algorithm
            temperature: Temperature for move selection (higher = more exploration)
            dirichlet_alpha: Alpha parameter for Dirichlet noise (for exploration)
            dirichlet_epsilon: Weight of Dirichlet noise in root node
            batch_size: Number of positions to evaluate in parallel on GPU (default 32)
            batch_evaluator: Optional shared BatchEvaluator. If provided, NN
                calls are routed through it so that leaves from many games
                running in parallel coalesce into one GPU call.
        """
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.batch_evaluator = batch_evaluator

    def search(self, game_state: HnefataflGame, temperature: Optional[float] = None,
               add_noise: bool = False) -> Tuple[Move, np.ndarray]:
        """
        Run MCTS from the given game state with batched neural network evaluation.

        Args:
            game_state: Current game state
            temperature: Temperature for move selection (overrides default if provided)
            add_noise: If True, add Dirichlet noise to root priors (self-play only)

        Returns:
            Tuple of (best_move, move_probabilities)
            - best_move: Best move according to MCTS
            - move_probabilities: Probability distribution over moves
        """
        # Use provided temperature or default
        if temperature is not None:
            old_temperature = self.temperature
            self.temperature = temperature

        root = MCTSNode(game_state)

        # Pre-expand the root so Dirichlet noise (if enabled) is applied before
        # any simulation selects a child. Adds one NN eval per move.
        if not root.game_state.is_game_over():
            root_legal = root.game_state.get_legal_moves()
            if root_legal:
                root_policies, root_values = self._evaluate_positions_batch([root])
                root.expand(root_policies[0], root_legal)
                if add_noise:
                    add_dirichlet_noise(
                        root,
                        epsilon=self.dirichlet_epsilon,
                        alpha=self.dirichlet_alpha,
                    )
                self._backpropagate([root], float(root_values[0]), root.game_state.current_player)

        # Run simulations in batches for GPU efficiency.
        # Use the full configured batch_size — virtual loss keeps parallel
        # paths distinct, so larger mini-batches are fine and amortize the
        # per-call GPU launch overhead.
        mini_batch_size = min(self.batch_size, self.num_simulations)
        sims_completed = 0

        while sims_completed < self.num_simulations:
            # Determine how many simulations in this mini-batch
            current_batch_size = min(mini_batch_size, self.num_simulations - sims_completed)

            # Collect leaf nodes for batch evaluation
            leaf_nodes = []
            search_paths = []

            for _ in range(current_batch_size):
                node = root
                search_path = [node]

                # Selection: traverse tree using UCB until we reach a leaf
                while not node.is_leaf() and not node.game_state.is_game_over():
                    node = node.select_child(self.c_puct)
                    search_path.append(node)

                # Apply virtual loss to prevent other parallel searches from picking same path
                for path_node in search_path:
                    path_node.virtual_loss += 1

                leaf_nodes.append(node)
                search_paths.append(search_path)

            # Separate terminal and non-terminal nodes
            terminal_results = []
            non_terminal_nodes = []
            non_terminal_indices = []

            for i, node in enumerate(leaf_nodes):
                if node.game_state.is_game_over():
                    value = self._get_terminal_value(node.game_state, root.game_state.current_player)
                    terminal_results.append((i, value, None, None))
                else:
                    legal_moves = node.game_state.get_legal_moves()
                    if len(legal_moves) == 0:
                        terminal_results.append((i, -1.0, None, None))
                    else:
                        non_terminal_nodes.append(node)
                        non_terminal_indices.append(i)

            # Batch evaluate non-terminal positions (THIS is where GPU batching happens!)
            if non_terminal_nodes:
                policies, values = self._evaluate_positions_batch(non_terminal_nodes)

                # Expand nodes
                for idx, node in enumerate(non_terminal_nodes):
                    legal_moves = node.game_state.get_legal_moves()
                    node.expand(policies[idx], legal_moves)

            # Backpropagate all results
            for i, search_path in enumerate(search_paths):
                # Get value for this node
                if any(result[0] == i for result in terminal_results):
                    # Terminal node
                    value = next(result[1] for result in terminal_results if result[0] == i)
                else:
                    # Non-terminal node - find it in our batch results
                    batch_idx = non_terminal_indices.index(i)
                    value = values[batch_idx]

                # Remove virtual loss and backpropagate
                for path_node in search_path:
                    path_node.virtual_loss -= 1

                self._backpropagate(search_path, value, root.game_state.current_player)

            sims_completed += current_batch_size

        # Select move based on visit counts
        result = self._select_move(root)

        # Restore original temperature if it was overridden
        if temperature is not None:
            self.temperature = old_temperature

        return result

    def _evaluate_position(self, game_state: HnefataflGame) -> Tuple[np.ndarray, float]:
        """
        Evaluate single position using neural network (legacy method, prefer batch version).

        Returns:
            Tuple of (policy, value)
        """
        from hnefatafl.game import get_policy_size

        if self.neural_network is None:
            # Random policy if no network
            policy_size = get_policy_size()
            policy = np.ones(policy_size) / policy_size  # Uniform distribution
            value = 0.0
        else:
            # Use neural network
            import torch
            state_tensor = torch.FloatTensor(game_state.encode_state()).unsqueeze(0)

            model_device = next(self.neural_network.parameters()).device
            state_tensor = state_tensor.to(model_device)

            policy, value = self.neural_network.predict(state_tensor)
            policy = policy.cpu().numpy()

        return policy, value

    def _evaluate_positions_batch(self, nodes: List[MCTSNode]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Evaluate multiple positions in a single batched GPU call (MUCH faster!).

        Args:
            nodes: List of MCTSNode objects to evaluate

        Returns:
            Tuple of (policies, values) where each is a list matching input nodes
        """
        from hnefatafl.game import get_policy_size
        import torch
        import torch.nn.functional as F

        if not nodes:
            return [], []

        # If there's no evaluator AND no in-process model, fall back to a
        # uniform prior — only hit in tests that run MCTS without a network.
        if self.neural_network is None and self.batch_evaluator is None:
            policy_size = get_policy_size()
            policies = [np.ones(policy_size) / policy_size for _ in nodes]
            values = [0.0 for _ in nodes]
            return policies, values

        # Batch encode all game states
        states = np.array([node.game_state.encode_state() for node in nodes], dtype=np.float32)

        # Prefer the shared batcher (RemoteEvaluator or in-process
        # BatchEvaluator). Without one, fall back to the local model.
        if self.batch_evaluator is not None:
            policies_np, values_np = self.batch_evaluator.evaluate(states)
        else:
            state_tensor = torch.from_numpy(states).float()
            model_device = next(self.neural_network.parameters()).device
            state_tensor = state_tensor.to(model_device)
            self.neural_network.eval()
            with torch.no_grad():
                policy_logits, value_tensor = self.neural_network.forward(state_tensor)
                policy_probs = F.softmax(policy_logits, dim=1)
            policies_np = policy_probs.cpu().numpy()
            values_np = value_tensor.cpu().numpy().flatten()

        policies = [policies_np[i] for i in range(len(nodes))]
        values = [float(values_np[i]) for i in range(len(nodes))]

        return policies, values

    def _get_terminal_value(self, game_state: HnefataflGame, root_player: Player) -> float:
        """
        Get value for terminal game state.

        Returns:
            1.0 if root player won, -1.0 if root player lost, 0.0 for draw
        """
        winner = game_state.get_winner()

        if winner is None:
            return 0.0  # Draw
        elif winner == root_player:
            return 1.0  # Win for root player
        else:
            return -1.0  # Loss for root player

    def _backpropagate(self, search_path: List[MCTSNode], value: float, root_player: Player):
        """
        Backpropagate value through the search path.

        Args:
            search_path: List of nodes from root to leaf
            value: Value to backpropagate
            root_player: Player at the root node
        """
        for node in reversed(search_path):
            # Flip value if we're backpropagating to opponent's node
            if node.game_state.current_player != root_player:
                node_value = -value
            else:
                node_value = value

            node.update(node_value)

    def _select_move(self, root: MCTSNode) -> Tuple[Move, np.ndarray]:
        """
        Select move from root based on visit counts.

        Args:
            root: Root node of the search tree

        Returns:
            Tuple of (best_move, move_probabilities)
            - best_move: The selected move
            - move_probabilities: Fixed-size policy vector of size 4840
        """
        from hnefatafl.game import encode_move, get_policy_size

        visit_counts = []
        moves = []

        for move, child in root.children.items():
            moves.append(move)
            visit_counts.append(child.visit_count)

        visit_counts = np.array(visit_counts)

        # Apply temperature
        if self.temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visit_counts)
            probs = np.zeros(len(visit_counts))
            probs[best_idx] = 1.0
        else:
            # Probabilistic selection based on visit counts
            visit_counts_temp = visit_counts ** (1.0 / self.temperature)
            probs = visit_counts_temp / np.sum(visit_counts_temp)

        # Select move
        if self.temperature == 0:
            best_move = moves[best_idx]
        else:
            best_idx = np.random.choice(len(moves), p=probs)
            best_move = moves[best_idx]

        # Create fixed-size probability distribution over ALL possible moves
        # This is crucial for training - all policy vectors must be the same size
        policy_size = get_policy_size()
        full_probs = np.zeros(policy_size)

        # Map the probabilities to their indices in the full policy vector
        for i, move in enumerate(moves):
            move_index = encode_move(move)
            full_probs[move_index] = probs[i]

        return best_move, full_probs

    def get_move_probabilities(self, game_state: HnefataflGame) -> Dict[Move, float]:
        """
        Get probability distribution over moves from current position.

        Returns:
            Dictionary mapping moves to probabilities
        """
        best_move, probs = self.search(game_state)
        legal_moves = game_state.get_legal_moves()

        move_probs = {}
        for i, move in enumerate(legal_moves):
            move_probs[move] = probs[i] if i < len(probs) else 0.0

        return move_probs


def add_dirichlet_noise(node: MCTSNode, epsilon: float = 0.25, alpha: float = 0.3):
    """
    Add Dirichlet noise to root node for exploration during self-play.

    Args:
        node: Root node to add noise to
        epsilon: Weight of noise
        alpha: Dirichlet alpha parameter
    """
    if not node.children:
        return

    noise = np.random.dirichlet([alpha] * len(node.children))

    for i, child in enumerate(node.children.values()):
        child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * noise[i]
