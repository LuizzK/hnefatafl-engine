"""
Monte Carlo Tree Search (MCTS) implementation for Copenhagen Hnefatafl

Uses PUCT (Predictor + Upper Confidence bounds applied to Trees) algorithm
with neural network guidance, similar to AlphaZero.
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple
from hnefatafl.game import HnefataflGame, Move, Player


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

        self.is_expanded = False

    def get_value(self) -> float:
        """Get average value (Q-value) of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

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

        Args:
            policy_probs: Policy probabilities from neural network
            legal_moves: List of legal moves from this position
        """
        if self.is_expanded:
            return

        for move in legal_moves:
            # Create child node
            child_game = self.game_state.copy()
            child_game.make_move(move)

            # Get prior probability for this move
            # For now, use uniform distribution (will be improved with proper move encoding)
            prior_prob = 1.0 / len(legal_moves)

            child_node = MCTSNode(child_game, parent=self, prior_prob=prior_prob, move=move)
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
                 c_puct: float = 1.5, temperature: float = 1.0):
        """
        Initialize MCTS.

        Args:
            neural_network: Neural network for position evaluation
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for PUCT algorithm
            temperature: Temperature for move selection (higher = more exploration)
        """
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

    def search(self, game_state: HnefataflGame) -> Tuple[Move, np.ndarray]:
        """
        Run MCTS from the given game state.

        Args:
            game_state: Current game state

        Returns:
            Tuple of (best_move, move_probabilities)
            - best_move: Best move according to MCTS
            - move_probabilities: Probability distribution over moves
        """
        root = MCTSNode(game_state)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree using UCB until we reach a leaf
            while not node.is_leaf() and not node.game_state.is_game_over():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            # Check if game is over
            if node.game_state.is_game_over():
                # Terminal node
                value = self._get_terminal_value(node.game_state, root.game_state.current_player)
            else:
                # Expansion and evaluation
                legal_moves = node.game_state.get_legal_moves()

                if len(legal_moves) == 0:
                    # No legal moves (shouldn't happen if game logic is correct)
                    value = -1.0
                else:
                    # Get neural network policy and value
                    policy, value = self._evaluate_position(node.game_state)

                    # Expand node
                    node.expand(policy, legal_moves)

            # Backpropagation
            self._backpropagate(search_path, value, root.game_state.current_player)

        # Select move based on visit counts
        return self._select_move(root)

    def _evaluate_position(self, game_state: HnefataflGame) -> Tuple[np.ndarray, float]:
        """
        Evaluate position using neural network.

        Returns:
            Tuple of (policy, value)
        """
        if self.neural_network is None:
            # Random policy if no network
            legal_moves = game_state.get_legal_moves()
            policy = np.ones(4400) / 4400  # Uniform distribution
            value = 0.0
        else:
            # Use neural network
            import torch
            state_tensor = torch.FloatTensor(game_state.encode_state()).unsqueeze(0)

            if torch.cuda.is_available():
                state_tensor = state_tensor.cuda()

            policy, value = self.neural_network.predict(state_tensor)
            policy = policy.cpu().numpy()

        return policy, value

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
        """
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

        # Create full probability distribution over all moves (for training)
        legal_moves = root.game_state.get_legal_moves()
        full_probs = np.zeros(len(legal_moves))

        for i, move in enumerate(legal_moves):
            if move in moves:
                idx = moves.index(move)
                full_probs[i] = probs[idx]

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
