"""
Tests for the centralized GPU inference server.

These tests pin the server to CPU so they run quickly on any machine.
They verify that RemoteEvaluator.evaluate() is numerically identical to
BatchEvaluator.evaluate() (same model, same inputs) and that the server
correctly scatters results across workers.
"""

import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from hnefatafl.game import HnefataflGame
from hnefatafl.infer_server import InferenceServer, RemoteEvaluator
from hnefatafl.mcts import BatchEvaluator
from hnefatafl.network import create_model


def _make_states(batch: int, seed: int = 0) -> np.ndarray:
    """Build a batch of plausible encoded states by playing random moves."""
    rng = np.random.default_rng(seed)
    states = []
    game = HnefataflGame()
    for _ in range(batch):
        if game.is_game_over():
            game = HnefataflGame()
        states.append(game.encode_state().astype(np.float32))
        legal = game.get_legal_moves()
        game.make_move(legal[rng.integers(len(legal))])
    return np.stack(states, axis=0)


def _make_model(seed: int = 42):
    torch.manual_seed(seed)
    model = create_model(num_channels=16, num_res_blocks=1, device='cpu')
    model.eval()
    return model


def test_single_worker_round_trip():
    """One request, one response — values match direct model.forward()."""
    model = _make_model()
    states = _make_states(4)

    ctx = mp.get_context('spawn')
    server = InferenceServer(model, num_workers=1, ctx=ctx, max_batch=32, max_wait_ms=2.0)
    server.start()
    try:
        evaluator = RemoteEvaluator(server.request_queue, server.response_queue(0), 0)
        policies, values = evaluator.evaluate(states)

        # Compare against direct path
        expected_p, expected_v = server.evaluate_direct(states)
        np.testing.assert_allclose(policies, expected_p, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(values, expected_v, rtol=1e-5, atol=1e-6)
        assert policies.shape[0] == 4
        assert values.shape == (4,)
    finally:
        server.stop()


def test_identical_output_to_batchevaluator():
    """Same model + inputs → bitwise-identical policies/values on CPU."""
    model = _make_model(seed=7)
    states = _make_states(5, seed=123)

    # Direct BatchEvaluator (in-process, threaded)
    be = BatchEvaluator(model, max_batch=32, max_wait_ms=2.0)
    try:
        be_policies, be_values = be.evaluate(states)
    finally:
        be.close()

    # RemoteEvaluator (cross-process server)
    ctx = mp.get_context('spawn')
    server = InferenceServer(model, num_workers=1, ctx=ctx, max_batch=32, max_wait_ms=2.0)
    server.start()
    try:
        evaluator = RemoteEvaluator(server.request_queue, server.response_queue(0), 0)
        rs_policies, rs_values = evaluator.evaluate(states)
    finally:
        server.stop()

    # CPU math is deterministic: results should be bitwise equal.
    np.testing.assert_array_equal(be_policies, rs_policies)
    np.testing.assert_array_equal(be_values, rs_values)


def test_batching_across_workers():
    """Multiple in-process workers sharing one server still produce correct results."""
    import threading

    model = _make_model(seed=3)
    num_workers = 4
    reqs_per_worker = 10

    ctx = mp.get_context('spawn')
    server = InferenceServer(model, num_workers=num_workers, ctx=ctx,
                             max_batch=128, max_wait_ms=5.0)
    server.start()

    # Per-worker fixed seed so we can recompute expected results.
    payloads = {wid: _make_states(3, seed=100 + wid) for wid in range(num_workers)}
    results = {wid: [] for wid in range(num_workers)}

    def worker_fn(wid):
        ev = RemoteEvaluator(server.request_queue, server.response_queue(wid), wid)
        for _ in range(reqs_per_worker):
            p, v = ev.evaluate(payloads[wid])
            results[wid].append((p.copy(), v.copy()))

    try:
        threads = [threading.Thread(target=worker_fn, args=(wid,)) for wid in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "worker hung"

        # Verify every response matches the direct-path expectation for that worker.
        # Tolerance is loose enough to absorb float reduction-order differences
        # from variable batch sizes on CPU softmax.
        for wid in range(num_workers):
            expected_p, expected_v = server.evaluate_direct(payloads[wid])
            for p, v in results[wid]:
                np.testing.assert_allclose(p, expected_p, rtol=1e-5, atol=1e-8)
                np.testing.assert_allclose(v, expected_v, rtol=1e-5, atol=1e-8)

        # Server should have processed some batches (>= one per call, often fewer).
        assert server.stats_batches >= 1
        assert server.stats_states == num_workers * reqs_per_worker * 3
    finally:
        server.stop()


def test_server_error_propagates_to_worker():
    """If the server's forward pass raises, the worker's evaluate() must raise (not hang)."""
    model = _make_model(seed=9)

    # Wrap model so forward() raises
    class BrokenModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            raise RuntimeError("synthetic forward failure")

        def parameters(self, recurse: bool = True):
            return self.inner.parameters(recurse=recurse)

    broken = BrokenModel(model)

    ctx = mp.get_context('spawn')
    server = InferenceServer(broken, num_workers=1, ctx=ctx,
                             max_batch=16, max_wait_ms=2.0)
    server.start()
    try:
        evaluator = RemoteEvaluator(server.request_queue, server.response_queue(0),
                                    0, timeout=10.0)
        states = _make_states(2, seed=0)
        with pytest.raises(RuntimeError, match="synthetic forward failure"):
            evaluator.evaluate(states)
    finally:
        server.stop()
