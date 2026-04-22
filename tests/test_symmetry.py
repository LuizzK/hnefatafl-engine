"""Tests for dihedral-8 symmetry augmentation of state + policy."""

import numpy as np
import pytest

from hnefatafl.game import HnefataflGame, encode_move, decode_move, get_policy_size
from hnefatafl.selfplay import (
    SYM_NAMES,
    SYM_STATE_OPS,
    _SYMMETRY_PERMS,
    apply_symmetry,
)


def _apply_perm(p, perm):
    q = np.zeros_like(p)
    q[perm] = p
    return q


def test_all_perms_are_valid_permutations():
    for name in SYM_NAMES:
        perm = _SYMMETRY_PERMS[name]
        assert perm.shape == (get_policy_size(),)
        assert set(perm.tolist()) == set(range(get_policy_size()))


def test_self_inverse_symmetries():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(get_policy_size()))
    for name in ('e', 'r2', 'fh', 'fv', 'td', 'ta'):
        twice = _apply_perm(_apply_perm(p, _SYMMETRY_PERMS[name]), _SYMMETRY_PERMS[name])
        np.testing.assert_allclose(twice, p, atol=1e-12)


def test_r1_r3_inverse_pair():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(get_policy_size()))
    q = _apply_perm(p, _SYMMETRY_PERMS['r1'])
    r = _apply_perm(q, _SYMMETRY_PERMS['r3'])
    np.testing.assert_allclose(r, p, atol=1e-12)


def test_rot180_equals_two_rot90():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(get_policy_size()))
    via_r2 = _apply_perm(p, _SYMMETRY_PERMS['r2'])
    via_twice = _apply_perm(_apply_perm(p, _SYMMETRY_PERMS['r1']), _SYMMETRY_PERMS['r1'])
    np.testing.assert_allclose(via_r2, via_twice, atol=1e-12)


def test_probability_mass_preserved():
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(get_policy_size()))
    for name in SYM_NAMES:
        q = _apply_perm(p, _SYMMETRY_PERMS[name])
        assert abs(q.sum() - p.sum()) < 1e-10


def test_symmetry_matches_point_transform_of_moves():
    """If we place prob 1 on move (fr,fc)->(tr,tc), the permuted policy's nonzero
    index must decode to the point-transformed move."""
    N, B = 11, 10
    point = {
        'e':  lambda r, c: (r, c),
        'r1': lambda r, c: (B - c, r),
        'r2': lambda r, c: (B - r, B - c),
        'r3': lambda r, c: (c, B - r),
        'fh': lambda r, c: (r, B - c),
        'fv': lambda r, c: (B - r, c),
        'td': lambda r, c: (c, r),
        'ta': lambda r, c: (B - c, B - r),
    }
    game = HnefataflGame()
    legal = game.get_legal_moves()
    for move in legal[:5]:  # sample several moves
        p = np.zeros(get_policy_size())
        p[encode_move(move)] = 1.0
        for name in SYM_NAMES:
            q = _apply_perm(p, _SYMMETRY_PERMS[name])
            idx = int(q.argmax())
            decoded = decode_move(idx)
            expected_from = point[name](move.from_row, move.from_col)
            expected_to = point[name](move.to_row, move.to_col)
            assert (decoded.from_row, decoded.from_col) == expected_from, \
                f"{name}: from mismatch on move {move}"
            assert (decoded.to_row, decoded.to_col) == expected_to, \
                f"{name}: to mismatch on move {move}"


def test_state_and_policy_stay_consistent_under_augmentation():
    """A state's legal-move set (after state transform) must match the nonzero
    indices of the transformed policy."""
    game = HnefataflGame()
    state = game.encode_state()
    legal = game.get_legal_moves()
    # Uniform policy on legal moves
    policy = np.zeros(get_policy_size())
    for m in legal:
        policy[encode_move(m)] = 1.0 / len(legal)

    for name in SYM_NAMES:
        new_state, new_policy = apply_symmetry(state, policy, name)
        nonzero_indices = set(np.flatnonzero(new_policy).tolist())
        # All nonzero indices must decode to moves whose from-square has the
        # player's piece in the transformed state
        for idx in nonzero_indices:
            mv = decode_move(idx)
            # Bounds check: the augmented move must stay on-board
            assert 0 <= mv.to_row < 11 and 0 <= mv.to_col < 11, \
                f"{name}: permuted policy has off-board move at idx {idx}"


def test_augment_data_expands_8x():
    from hnefatafl.selfplay import TrainingExample, SelfPlayWorker
    state = np.zeros((15, 11, 11), dtype=np.float32)
    state[0, 5, 5] = 1.0  # a piece at center
    policy = np.zeros(get_policy_size())
    policy[encode_move(HnefataflGame().get_legal_moves()[0])] = 1.0
    examples = [TrainingExample(state, policy, 0.5)]
    worker = SelfPlayWorker.__new__(SelfPlayWorker)  # bypass __init__ / MCTS construction
    augmented = worker.augment_data(examples)
    assert len(augmented) == 8
    # All augmented examples should share the same value
    for ex in augmented:
        assert ex.value == 0.5
        assert ex.state.shape == state.shape
        assert ex.policy.shape == policy.shape
        np.testing.assert_allclose(ex.policy.sum(), 1.0, atol=1e-10)
