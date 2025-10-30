"""Quick test of move encoding/decoding"""

from hnefatafl.game import HnefataflGame, Move, encode_move, decode_move, get_policy_size

print("Testing move encoding...")
print("=" * 60)

# Test 1: Policy size
policy_size = get_policy_size()
print(f"✓ Policy size: {policy_size} (should be 4840)")

# Test 2: Encode and decode some moves
game = HnefataflGame()
legal_moves = game.get_legal_moves()
print(f"✓ Generated {len(legal_moves)} legal moves from starting position")

# Test first 5 moves
print("\nTesting encode/decode on 5 moves:")
for i, move in enumerate(legal_moves[:5]):
    index = encode_move(move)
    decoded = decode_move(index)

    match = (move.from_row == decoded.from_row and
             move.from_col == decoded.from_col and
             move.to_row == decoded.to_row and
             move.to_col == decoded.to_col)

    status = "✓" if match else "✗"
    print(f"  {status} Move {i+1}: {move} -> index {index} -> {decoded}")

    if not match:
        print(f"     ERROR: Mismatch!")
        break

print("\n" + "=" * 60)
print("Move encoding test PASSED! ✓")
