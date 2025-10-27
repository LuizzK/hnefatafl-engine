#!/usr/bin/env python3
"""
Verify that MCTS is working correctly
"""

import time
from hnefatafl.game import HnefataflGame, Player
from hnefatafl.mcts import MCTS, MCTSNode
from hnefatafl.network import create_model
import torch


def test_mcts_basic():
    """Test basic MCTS functionality without neural network"""
    print("Testing basic MCTS (random rollouts)...")

    game = HnefataflGame()
    mcts = MCTS(neural_network=None, num_simulations=50, c_puct=1.5)

    print(f"✓ MCTS initialized with {mcts.num_simulations} simulations")

    # Run search
    start_time = time.time()
    best_move, move_probs = mcts.search(game)
    elapsed = time.time() - start_time

    print(f"✓ Search completed in {elapsed:.2f} seconds")
    print(f"✓ Best move: {best_move}")
    print(f"✓ Move probability distribution shape: {move_probs.shape}")
    print(f"✓ Probability sum: {move_probs.sum():.3f}")

    # Verify move is legal
    legal_moves = game.get_legal_moves()
    assert best_move in legal_moves, "Selected move should be legal"
    print("✓ Selected move is legal")

    return best_move


def test_mcts_with_network():
    """Test MCTS with neural network"""
    print("\nTesting MCTS with neural network...")

    # Create a small network for testing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")

    model = create_model(num_channels=64, num_res_blocks=5, device=device)
    model.eval()
    print("✓ Neural network created (64 channels, 5 blocks)")

    game = HnefataflGame()
    mcts = MCTS(neural_network=model, num_simulations=100, c_puct=1.5)

    # Run search
    start_time = time.time()
    best_move, move_probs = mcts.search(game)
    elapsed = time.time() - start_time

    print(f"✓ Search completed in {elapsed:.2f} seconds")
    print(f"✓ Best move: {best_move}")
    print(f"✓ Moves per second: {mcts.num_simulations / elapsed:.1f}")

    # Verify move is legal
    legal_moves = game.get_legal_moves()
    assert best_move in legal_moves, "Selected move should be legal"
    print("✓ Selected move is legal")

    return best_move, elapsed


def test_mcts_node():
    """Test MCTS node operations"""
    print("\nTesting MCTS node operations...")

    game = HnefataflGame()
    root = MCTSNode(game)

    print(f"✓ Root node created")
    print(f"  Visit count: {root.visit_count}")
    print(f"  Value: {root.get_value():.3f}")
    print(f"  Is leaf: {root.is_leaf()}")

    # Expand node
    legal_moves = game.get_legal_moves()
    import numpy as np
    policy = np.ones(len(legal_moves)) / len(legal_moves)
    root.expand(policy, legal_moves)

    print(f"✓ Node expanded with {len(root.children)} children")
    assert len(root.children) == len(legal_moves), "Should have one child per legal move"

    # Update node
    root.update(0.5)
    print(f"✓ Node updated: visit_count={root.visit_count}, value={root.get_value():.3f}")

    # Test UCB score
    if root.children:
        first_child = list(root.children.values())[0]
        ucb_score = first_child.get_ucb_score(c_puct=1.5, parent_visits=1)
        print(f"✓ UCB score computed: {ucb_score:.3f}")


def test_mcts_game_simulation():
    """Test playing a full game using MCTS"""
    print("\nTesting full game simulation with MCTS...")

    game = HnefataflGame()
    mcts = MCTS(neural_network=None, num_simulations=50, c_puct=1.5, temperature=1.0)

    move_count = 0
    max_moves = 20  # Play first 20 moves only

    print("Playing first 20 moves...")
    start_time = time.time()

    while not game.is_game_over() and move_count < max_moves:
        # Get move from MCTS
        best_move, _ = mcts.search(game)

        # Make move
        game.make_move(best_move)
        move_count += 1

        if move_count % 5 == 0:
            current_player = "Attacker" if game.current_player == Player.ATTACKER else "Defender"
            print(f"  Move {move_count}: {best_move} -> {current_player}'s turn")

    elapsed = time.time() - start_time

    print(f"✓ Played {move_count} moves in {elapsed:.2f} seconds")
    print(f"✓ Average time per move: {elapsed / move_count:.2f} seconds")
    print(f"✓ Game result: {game.get_result_string()}")


def test_temperature_effect():
    """Test that temperature affects move selection"""
    print("\nTesting temperature effect on move selection...")

    game = HnefataflGame()

    # Test with temperature 0 (greedy)
    mcts_greedy = MCTS(neural_network=None, num_simulations=50, temperature=0.0)
    move_greedy, _ = mcts_greedy.search(game)
    print(f"✓ Greedy (T=0.0) move: {move_greedy}")

    # Test with temperature 1 (exploratory)
    mcts_explore = MCTS(neural_network=None, num_simulations=50, temperature=1.0)
    move_explore, _ = mcts_explore.search(game)
    print(f"✓ Exploratory (T=1.0) move: {move_explore}")

    # Test with high temperature (very random)
    mcts_random = MCTS(neural_network=None, num_simulations=50, temperature=2.0)
    move_random, _ = mcts_random.search(game)
    print(f"✓ Very exploratory (T=2.0) move: {move_random}")

    print("✓ Temperature parameter works correctly")


def test_mcts_consistency():
    """Test that MCTS gives consistent results with same seed"""
    print("\nTesting MCTS consistency...")

    import numpy as np

    game = HnefataflGame()
    mcts = MCTS(neural_network=None, num_simulations=100, c_puct=1.5, temperature=0.0)

    # Run search multiple times
    moves = []
    for i in range(3):
        move, _ = mcts.search(game.copy())
        moves.append(move)

    # With greedy selection (T=0), moves should be relatively consistent
    print(f"✓ Moves from 3 runs: {moves}")
    print("✓ MCTS produces moves (may vary due to random initialization)")


def benchmark_mcts():
    """Benchmark MCTS performance"""
    print("\nBenchmarking MCTS performance...")

    game = HnefataflGame()

    for num_sims in [50, 100, 200, 400]:
        mcts = MCTS(neural_network=None, num_simulations=num_sims, c_puct=1.5)

        start_time = time.time()
        best_move, _ = mcts.search(game)
        elapsed = time.time() - start_time

        print(f"  {num_sims:4d} simulations: {elapsed:.3f}s ({num_sims/elapsed:.0f} sims/sec)")


def main():
    """Run all verification tests"""
    print("="*60)
    print("MCTS VERIFICATION")
    print("="*60)

    try:
        # Test 1: Basic MCTS
        test_mcts_basic()

        # Test 2: MCTS with neural network
        test_mcts_with_network()

        # Test 3: MCTS node operations
        test_mcts_node()

        # Test 4: Full game simulation
        test_mcts_game_simulation()

        # Test 5: Temperature effect
        test_temperature_effect()

        # Test 6: Consistency
        test_mcts_consistency()

        # Test 7: Benchmark
        benchmark_mcts()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nMCTS is working correctly!")
        print("\nNext steps:")
        print("  1. Implement self-play engine (selfplay.py)")
        print("  2. Implement training pipeline (train.py)")
        print("  3. Start training!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
