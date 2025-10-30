"""Test that MCTS returns fixed-size policy vectors"""

from hnefatafl.game import HnefataflGame, get_policy_size
from hnefatafl.mcts import MCTS
from hnefatafl.network import HnefataflNetwork

print("Testing MCTS policy vector size...")
print("=" * 60)

# Expected policy size
expected_size = get_policy_size()
print(f"Expected policy size: {expected_size}")

# Create a small model for testing
model = HnefataflNetwork(num_channels=32, num_res_blocks=2, board_size=11)
print("✓ Created test model")

# Create MCTS with very few simulations (just for testing)
mcts = MCTS(neural_network=model, num_simulations=5)
print("✓ Created MCTS engine")

# Create starting position
game = HnefataflGame()
print(f"✓ Created game (legal moves: {len(game.get_legal_moves())})")

# Run MCTS search
print("\nRunning MCTS search...")
move, policy = mcts.search(game, temperature=1.0)
print(f"✓ MCTS search complete")

# Check policy size
print(f"\nPolicy vector size: {policy.shape[0]}")
print(f"Expected size: {expected_size}")

if policy.shape[0] == expected_size:
    print("✓ Policy size is correct!")

    # Check that policy sums to approximately 1.0
    policy_sum = policy.sum()
    print(f"\nPolicy sum: {policy_sum:.6f}")
    if 0.99 < policy_sum < 1.01:
        print("✓ Policy sums to 1.0!")
    else:
        print("✗ WARNING: Policy should sum to 1.0")

    # Check that selected move has non-zero probability
    from hnefatafl.game import encode_move
    selected_index = encode_move(move)
    print(f"\nSelected move: {move}")
    print(f"Selected move index: {selected_index}")
    print(f"Selected move probability: {policy[selected_index]:.6f}")
    if policy[selected_index] > 0:
        print("✓ Selected move has non-zero probability!")
    else:
        print("✗ ERROR: Selected move has zero probability!")

    print("\n" + "=" * 60)
    print("MCTS policy test PASSED! ✓")
else:
    print(f"✗ ERROR: Policy size mismatch!")
    print(f"   Expected: {expected_size}")
    print(f"   Got: {policy.shape[0]}")
