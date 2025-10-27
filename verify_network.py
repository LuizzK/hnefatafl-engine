#!/usr/bin/env python3
"""
Verify that the neural network is working correctly
"""

import torch
import numpy as np
from hnefatafl.network import create_model, HnefataflNetwork, PolicyValueLoss
from hnefatafl.game import HnefataflGame


def test_network_creation():
    """Test that network can be created"""
    print("Testing network creation...")
    model = create_model(num_channels=128, num_res_blocks=10)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {param_count:,} parameters")
    print(f"✓ Device: {next(model.parameters()).device}")

    return model


def test_forward_pass(model):
    """Test forward pass through the network"""
    print("\nTesting forward pass...")

    # Create a dummy input (batch of 4 game states)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 15, 11, 11)

    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    # Forward pass
    policy_logits, value = model(dummy_input)

    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Policy output shape: {policy_logits.shape}")
    print(f"✓ Value output shape: {value.shape}")
    print(f"✓ Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")

    # Check that value is in valid range [-1, 1]
    assert value.min() >= -1.0 and value.max() <= 1.0, "Value should be in [-1, 1]"
    print("✓ Value predictions in valid range [-1, 1]")

    return policy_logits, value


def test_with_real_game_state(model):
    """Test network with real game state"""
    print("\nTesting with real game state...")

    game = HnefataflGame()
    state = game.encode_state()

    print(f"✓ Game state encoded: shape {state.shape}")

    # Convert to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    if torch.cuda.is_available():
        state_tensor = state_tensor.cuda()

    # Get prediction
    policy_probs, value = model.predict(state_tensor)

    print(f"✓ Policy distribution shape: {policy_probs.shape}")
    print(f"✓ Policy sum: {policy_probs.sum().item():.3f} (should be ~1.0)")
    print(f"✓ Value prediction: {value:.3f}")
    print(f"✓ Top 5 policy probabilities: {torch.topk(policy_probs, 5).values.cpu().numpy()}")


def test_loss_computation():
    """Test loss computation"""
    print("\nTesting loss computation...")

    batch_size = 8
    num_actions = 4840  # From network output: 121 * 4 * 10

    # Create dummy predictions and targets
    policy_logits = torch.randn(batch_size, num_actions)
    value_pred = torch.randn(batch_size, 1)

    # Create dummy targets
    target_policy = torch.softmax(torch.randn(batch_size, num_actions), dim=1)
    target_value = torch.rand(batch_size, 1) * 2 - 1  # Random values in [-1, 1]

    # Compute loss
    loss_fn = PolicyValueLoss()
    total_loss, policy_loss, value_loss = loss_fn(
        policy_logits, value_pred, target_policy, target_value
    )

    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ Policy loss: {policy_loss.item():.4f}")
    print(f"✓ Value loss: {value_loss.item():.4f}")

    # Check that loss is a scalar and not NaN
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    print("✓ Loss computation successful")


def test_backward_pass(model):
    """Test backward pass (gradient computation)"""
    print("\nTesting backward pass...")

    # Create dummy input and targets
    dummy_input = torch.randn(2, 15, 11, 11)
    target_policy = torch.softmax(torch.randn(2, 4840), dim=1)
    target_value = torch.rand(2, 1) * 2 - 1

    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        target_policy = target_policy.cuda()
        target_value = target_value.cuda()

    # Forward pass
    policy_logits, value_pred = model(dummy_input)

    # Compute loss
    loss_fn = PolicyValueLoss()
    total_loss, _, _ = loss_fn(policy_logits, value_pred, target_policy, target_value)

    # Backward pass
    total_loss.backward()

    # Check that gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "Gradients should be computed"
    print("✓ Backward pass successful")
    print("✓ Gradients computed correctly")


def test_model_save_load(model):
    """Test saving and loading model"""
    print("\nTesting model save/load...")

    from hnefatafl.network import save_checkpoint, load_checkpoint
    import tempfile
    import os

    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    save_checkpoint(model, optimizer, epoch=42, path=temp_path)
    print(f"✓ Model saved to {temp_path}")

    # Load model
    loaded_model, epoch = load_checkpoint(temp_path)
    print(f"✓ Model loaded from checkpoint (epoch {epoch})")

    # Verify loaded model works
    dummy_input = torch.randn(1, 15, 11, 11)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    policy, value = loaded_model(dummy_input)
    print("✓ Loaded model can perform inference")

    # Clean up
    os.remove(temp_path)
    print("✓ Checkpoint file cleaned up")


def main():
    """Run all verification tests"""
    print("="*60)
    print("NEURAL NETWORK VERIFICATION")
    print("="*60)

    try:
        # Test 1: Create model
        model = test_network_creation()

        # Test 2: Forward pass
        test_forward_pass(model)

        # Test 3: Real game state
        test_with_real_game_state(model)

        # Test 4: Loss computation
        test_loss_computation()

        # Test 5: Backward pass
        test_backward_pass(model)

        # Test 6: Save/load
        test_model_save_load(model)

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe neural network is working correctly!")
        print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

        if not torch.cuda.is_available():
            print("\n⚠️  Warning: No GPU detected. Training will be slow.")
            print("   Consider using a machine with NVIDIA GPU for faster training.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
