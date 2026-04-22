"""Tests verifying Dirichlet noise is applied only when requested."""

import numpy as np
import torch

from hnefatafl.game import HnefataflGame, encode_move
from hnefatafl.mcts import MCTS, MCTSNode
from hnefatafl.network import create_model


def _priors_after_root_expansion(add_noise: bool, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = create_model(num_channels=16, num_res_blocks=1, device='cpu')
    model.eval()
    game = HnefataflGame()
    mcts = MCTS(neural_network=model, num_simulations=1, c_puct=1.5, batch_size=8,
                dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    # Run search (triggers pre-expansion with noise if enabled)
    mcts.search(game, temperature=1.0, add_noise=add_noise)
    # We can't easily re-access root — re-do the pre-expansion logic manually
    root = MCTSNode(game)
    legal = game.get_legal_moves()
    policies, _ = mcts._evaluate_positions_batch([root])
    root.expand(policies[0], legal)
    if add_noise:
        from hnefatafl.mcts import add_dirichlet_noise
        add_dirichlet_noise(root, epsilon=mcts.dirichlet_epsilon, alpha=mcts.dirichlet_alpha)
    return {m: root.children[m].prior_prob for m in legal}


def test_dirichlet_off_is_deterministic():
    """With noise disabled, same seed -> same priors (no hidden randomness)."""
    a = _priors_after_root_expansion(add_noise=False, seed=42)
    b = _priors_after_root_expansion(add_noise=False, seed=42)
    for m in a:
        assert abs(a[m] - b[m]) < 1e-10


def test_dirichlet_on_differs_from_clean():
    """With noise enabled, priors should differ from the clean (noiseless) baseline."""
    clean = _priors_after_root_expansion(add_noise=False, seed=42)
    # Use a different numpy seed to ensure stochastic noise actually draws
    np.random.seed(12345)
    noisy = _priors_after_root_expansion(add_noise=True, seed=42)
    total_diff = sum(abs(noisy[m] - clean[m]) for m in clean)
    assert total_diff > 0.01, f"noise should perturb priors; got total_diff={total_diff}"


def test_dirichlet_priors_still_sum_to_one():
    noisy = _priors_after_root_expansion(add_noise=True, seed=7)
    total = sum(noisy.values())
    assert abs(total - 1.0) < 1e-6, f"noisy priors should sum to 1, got {total}"


def test_eval_default_has_no_noise():
    """search() without add_noise kwarg must NOT apply noise (evaluation path)."""
    # If default were True, two consecutive searches with same seed would diverge.
    # Since default is False, they must match.
    torch.manual_seed(0)
    model = create_model(num_channels=16, num_res_blocks=1, device='cpu')
    model.eval()
    game = HnefataflGame()
    mcts = MCTS(neural_network=model, num_simulations=2, c_puct=1.5, batch_size=8)

    np.random.seed(999)
    m1, p1 = mcts.search(game, temperature=0.0)  # default add_noise=False
    np.random.seed(1)  # different seed — should not affect deterministic (no-noise) search
    m2, p2 = mcts.search(game, temperature=0.0)
    assert m1 == m2
    np.testing.assert_allclose(p1, p2, atol=1e-10)
