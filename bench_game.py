"""Benchmark game-logic hot paths. Measures what self-play actually does
per MCTS simulation: copy -> make_move -> get_legal_moves -> encode_state.

Run before and after perf fixes to measure impact honestly.
"""

import time
import numpy as np
from hnefatafl.game import HnefataflGame


def bench(label, fn, warmup=50, iters=1000):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) * 1e6 / iters
    print(f"  {label:<40} {dt:8.2f} us/op")
    return dt


def make_midgame(moves=20):
    rng = np.random.default_rng(0)
    game = HnefataflGame()
    for _ in range(moves):
        legal = game.get_legal_moves()
        if not legal:
            break
        game.make_move(legal[rng.integers(len(legal))], _assume_legal=True)
    return game


def main():
    print("=" * 60)
    print("Game engine benchmarks")
    print("=" * 60)

    game_init = HnefataflGame()
    game_mid = make_midgame(20)

    print("\nInitial position:")
    bench("get_legal_moves()", lambda: game_init.get_legal_moves())
    bench("copy()", lambda: game_init.copy())
    bench("encode_state()", lambda: game_init.encode_state())

    print("\nMid-game position (~20 moves in):")
    bench("get_legal_moves()", lambda: game_mid.get_legal_moves())
    bench("copy()", lambda: game_mid.copy())
    bench("encode_state()", lambda: game_mid.encode_state())

    print("\nMCTS-like inner loop (copy + make_move + legal + encode):")
    def mcts_step():
        g = game_mid.copy()
        legal = g.get_legal_moves()
        g.make_move(legal[0], _assume_legal=True)
        g.encode_state()
    bench("1 sim step", mcts_step, iters=500)

    print("\nFull random game (~200 moves):")
    def full_game():
        g = HnefataflGame()
        rng = np.random.default_rng(42)
        while not g.is_game_over():
            legal = g.get_legal_moves()
            if not legal:
                break
            g.make_move(legal[rng.integers(len(legal))])
            if len(g.move_history) >= 200:
                break
    t0 = time.perf_counter()
    full_game()
    print(f"  one full game                             {(time.perf_counter() - t0) * 1e3:8.1f} ms")


if __name__ == "__main__":
    main()
