"""
Centralized GPU inference server for multi-process self-play.

Architecture:
- One `InferenceServer` runs in the main (trainer) process and owns the
  GPU and the model. It drains a shared request queue, batches pending
  requests up to `max_batch`, runs a single `model.forward()`, and
  scatters (policies, values) back to each requester's dedicated
  response queue.
- Each worker process holds a `RemoteEvaluator` which mirrors the
  `BatchEvaluator.evaluate(states_np) -> (policies_np, values_np)` API
  exactly, so MCTS code is unchanged.

Why this exists: without a server, each worker process submits its own
tiny GPU call serialized by the CUDA driver's queue, and the model
weights are pickled once per iteration per worker. With a server, the
GPU sees real batches and weights never leave the main process.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Sentinel that tells the server's drain thread to exit.
_STOP = "__STOP__"


@dataclass
class _Request:
    worker_id: int
    req_id: int
    states: np.ndarray  # (N, 15, 11, 11) float32


class RemoteEvaluator:
    """
    Worker-side proxy that mirrors `BatchEvaluator.evaluate()` exactly.

    Sends the state batch to the server via `request_queue` and blocks
    on `response_queue` until the server replies.
    """

    def __init__(self, request_queue, response_queue, worker_id: int,
                 timeout: float = 120.0):
        self._req_q = request_queue
        self._resp_q = response_queue
        self._worker_id = worker_id
        self._timeout = timeout
        self._counter = 0

    def evaluate(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Submit a (N, 15, 11, 11) batch. Blocks until server replies."""
        self._counter += 1
        req_id = self._counter
        states = np.ascontiguousarray(states, dtype=np.float32)
        self._req_q.put(_Request(self._worker_id, req_id, states))

        # Server sends back (req_id, policies, values) or (req_id, "ERR", msg)
        resp_req_id, payload_a, payload_b = self._resp_q.get(timeout=self._timeout)
        if resp_req_id != req_id:
            raise RuntimeError(
                f"RemoteEvaluator response out of order: expected {req_id}, got {resp_req_id}"
            )
        if isinstance(payload_a, str) and payload_a == "ERR":
            raise RuntimeError(f"InferenceServer error: {payload_b}")
        return payload_a, payload_b

    # Mirror BatchEvaluator API so MCTS's duck-typed check works uniformly.
    def close(self):
        pass


class InferenceServer:
    """
    Main-process server that batches NN calls from many workers.

    Lifecycle:
        server = InferenceServer(model, num_workers=N, ctx=mp_ctx)
        req_q = server.request_queue
        for wid in range(N):
            resp_q = server.response_queue(wid)
            # pass (req_q, resp_q, wid) to each worker's init
        server.start()
        ... run self-play ...
        server.stop()

    Each worker is assigned a unique worker_id in [0, num_workers) and
    must use the response_queue returned by `response_queue(worker_id)`.
    """

    def __init__(self, model, num_workers: int, ctx,
                 max_batch: int = 512, max_wait_ms: float = 2.0,
                 device: Optional[str] = None):
        import torch
        self._torch = torch
        self._F = __import__('torch.nn.functional', fromlist=['F'])

        self.model = model
        self.model.eval()
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        self.num_workers = num_workers
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0

        self._request_q = ctx.Queue()
        self._response_qs = [ctx.Queue() for _ in range(num_workers)]

        self._alive = False
        self._thread: Optional[threading.Thread] = None

        # Telemetry (server-side only; workers don't see these).
        self.stats_batches = 0
        self.stats_states = 0
        self.stats_forwards = 0
        self.stats_errors = 0

    @property
    def request_queue(self):
        return self._request_q

    def response_queue(self, worker_id: int):
        return self._response_qs[worker_id]

    def start(self):
        if self._alive:
            return
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="InferenceServer")
        self._thread.start()

    def stop(self, drain_timeout: float = 5.0):
        if not self._alive:
            return
        self._alive = False
        # Wake the drain loop by dropping a sentinel on the queue.
        try:
            self._request_q.put(_STOP)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=drain_timeout)
            self._thread = None

    def _collect_batch(self) -> list:
        """Block for one request, then drain any queued extras up to max_batch.

        Returns an empty list iff a stop sentinel was received.
        """
        # Block until the first request arrives (or a sentinel).
        first = self._request_q.get()
        if first is _STOP or not self._alive:
            return []

        batch = [first]
        deadline = time.monotonic() + self.max_wait

        # Pull more without blocking until we hit max_batch or the deadline.
        while len(batch) < self.max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._request_q.get(timeout=remaining)
            except Exception:
                break
            if item is _STOP:
                # Re-inject so outer loop sees the stop after flushing.
                self._alive = False
                break
            batch.append(item)
        return batch

    def _run(self):
        torch = self._torch
        F = self._F
        while self._alive:
            try:
                reqs = self._collect_batch()
            except Exception:
                if not self._alive:
                    break
                continue
            if not reqs:
                break

            try:
                all_states = np.concatenate([r.states for r in reqs], axis=0)
                state_tensor = torch.from_numpy(all_states).to(
                    self.device, non_blocking=True
                )
                with torch.no_grad():
                    logits, values = self.model.forward(state_tensor)
                    probs = F.softmax(logits, dim=1)
                policies_np = probs.detach().cpu().numpy()
                values_np = values.detach().cpu().numpy().flatten()

                self.stats_batches += 1
                self.stats_states += all_states.shape[0]
                self.stats_forwards += 1
            except Exception as exc:
                self.stats_errors += 1
                err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                # Attempt a CUDA recovery once for transient launch failures.
                recovered = False
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    # Best-effort retry
                    all_states = np.concatenate([r.states for r in reqs], axis=0)
                    state_tensor = torch.from_numpy(all_states).to(
                        self.device, non_blocking=True
                    )
                    with torch.no_grad():
                        logits, values = self.model.forward(state_tensor)
                        probs = F.softmax(logits, dim=1)
                    policies_np = probs.detach().cpu().numpy()
                    values_np = values.detach().cpu().numpy().flatten()
                    recovered = True
                except Exception:
                    recovered = False

                if not recovered:
                    # Fail fast: notify every waiting worker so they raise
                    # rather than hang forever on `response_queue.get()`.
                    for r in reqs:
                        try:
                            self._response_qs[r.worker_id].put((r.req_id, "ERR", err_msg))
                        except Exception:
                            pass
                    continue

            offset = 0
            for r in reqs:
                n = r.states.shape[0]
                p = policies_np[offset:offset + n]
                v = values_np[offset:offset + n]
                offset += n
                try:
                    self._response_qs[r.worker_id].put((r.req_id, p, v))
                except Exception:
                    # If a worker's queue is broken there's nothing to do.
                    pass

    # Convenience for tests / in-process parity with BatchEvaluator.
    def evaluate_direct(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run one forward pass synchronously, bypassing the queue.

        Used by tests and by code paths that want the server's model
        without spawning workers (e.g., main-process evaluation).
        """
        torch = self._torch
        F = self._F
        states = np.ascontiguousarray(states, dtype=np.float32)
        state_tensor = torch.from_numpy(states).to(self.device)
        with torch.no_grad():
            logits, values = self.model.forward(state_tensor)
            probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy(), values.detach().cpu().numpy().flatten()
