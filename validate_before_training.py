#!/usr/bin/env python3
"""
Pre-training validation script.

This script performs comprehensive checks to catch bugs BEFORE
you spend money on GPU training. Run this before every training session!

Usage:
    python validate_before_training.py
"""

import sys
import torch
import numpy as np
from typing import List, Tuple

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset}  {test_name}")
    if details:
        print(f"        {details}")


def validate_imports() -> bool:
    """Validate all imports work"""
    print_header("1. VALIDATING IMPORTS")

    all_passed = True

    try:
        from hnefatafl.game import HnefataflGame, Move, Player, Piece
        print_result("Game module", True)
    except Exception as e:
        print_result("Game module", False, str(e))
        all_passed = False

    try:
        from hnefatafl.network import HnefataflNetwork, create_model
        print_result("Network module", True)
    except Exception as e:
        print_result("Network module", False, str(e))
        all_passed = False

    try:
        from hnefatafl.mcts import MCTS
        print_result("MCTS module", True)
    except Exception as e:
        print_result("MCTS module", False, str(e))
        all_passed = False

    try:
        from hnefatafl.selfplay import SelfPlayWorker
        print_result("Self-play module", True)
    except Exception as e:
        print_result("Self-play module", False, str(e))
        all_passed = False

    try:
        from hnefatafl.train import Trainer, TrainingConfig
        print_result("Training module", True)
    except Exception as e:
        print_result("Training module", False, str(e))
        all_passed = False

    try:
        from config import get_config
        print_result("Config module", True)
    except Exception as e:
        print_result("Config module", False, str(e))
        all_passed = False

    return all_passed


def validate_game_engine() -> bool:
    """Validate game engine works correctly"""
    print_header("2. VALIDATING GAME ENGINE")

    all_passed = True

    try:
        from hnefatafl.game import HnefataflGame, Move, Player

        # Create game
        game = HnefataflGame()
        print_result("Create game", True)

        # Check initial state
        if game.current_player == Player.ATTACKER:
            print_result("Initial player is ATTACKER", True)
        else:
            print_result("Initial player is ATTACKER", False, f"Got {game.current_player}")
            all_passed = False

        # Get legal moves
        moves = game.get_legal_moves()
        if len(moves) > 0:
            print_result("Generate legal moves", True, f"{len(moves)} moves")
        else:
            print_result("Generate legal moves", False, "No moves found")
            all_passed = False

        # Make a move
        game.make_move(moves[0])
        if game.current_player == Player.DEFENDER:
            print_result("Make move and switch players", True)
        else:
            print_result("Make move and switch players", False)
            all_passed = False

        # Encode state
        state = game.encode_state()
        if state.shape == (15, 11, 11):
            print_result("Encode state", True, f"Shape {state.shape}")
        else:
            print_result("Encode state", False, f"Wrong shape {state.shape}")
            all_passed = False

        # Copy game
        game_copy = game.copy()
        print_result("Copy game", True)

    except Exception as e:
        print_result("Game engine validation", False, str(e))
        all_passed = False

    return all_passed


def validate_neural_network() -> bool:
    """Validate neural network works"""
    print_header("3. VALIDATING NEURAL NETWORK")

    all_passed = True

    try:
        from hnefatafl.network import create_model
        from hnefatafl.game import HnefataflGame

        # Create small model
        model = create_model(num_channels=32, num_res_blocks=2, device='cpu')
        print_result("Create model", True)

        # Forward pass
        game = HnefataflGame()
        state = game.encode_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        policy_logits, value = model(state_tensor)

        if policy_logits.shape[0] == 1:
            print_result("Policy output shape", True, f"{policy_logits.shape}")
        else:
            print_result("Policy output shape", False, f"Wrong shape {policy_logits.shape}")
            all_passed = False

        if value.shape == (1, 1):
            print_result("Value output shape", True, f"{value.shape}")
        else:
            print_result("Value output shape", False, f"Wrong shape {value.shape}")
            all_passed = False

        # Check value is in valid range after tanh
        value_item = value.item()
        if -1.0 <= value_item <= 1.0:
            print_result("Value in valid range", True, f"{value_item:.3f}")
        else:
            print_result("Value in valid range", False, f"{value_item}")
            all_passed = False

    except Exception as e:
        print_result("Neural network validation", False, str(e))
        all_passed = False

    return all_passed


def validate_mcts() -> bool:
    """Validate MCTS works"""
    print_header("4. VALIDATING MCTS")

    all_passed = True

    try:
        from hnefatafl.mcts import MCTS
        from hnefatafl.game import HnefataflGame
        from hnefatafl.network import create_model

        # Create model and MCTS
        model = create_model(num_channels=32, num_res_blocks=2, device='cpu')
        mcts = MCTS(neural_network=model, num_simulations=10)
        print_result("Create MCTS", True)

        # Run search
        game = HnefataflGame()
        move, policy = mcts.search(game)

        if move is not None:
            print_result("MCTS search", True, f"Found move {move}")
        else:
            print_result("MCTS search", False, "No move returned")
            all_passed = False

        if policy is not None and len(policy) > 0:
            print_result("MCTS policy", True, f"Policy size {len(policy)}")
        else:
            print_result("MCTS policy", False)
            all_passed = False

        # Test temperature parameter
        move2, policy2 = mcts.search(game, temperature=0.0)
        print_result("MCTS temperature parameter", True)

    except Exception as e:
        print_result("MCTS validation", False, str(e))
        all_passed = False

    return all_passed


def validate_selfplay() -> bool:
    """Validate self-play works"""
    print_header("5. VALIDATING SELF-PLAY")

    all_passed = True

    try:
        from hnefatafl.selfplay import SelfPlayWorker
        from hnefatafl.network import create_model

        # Create model and worker
        model = create_model(num_channels=32, num_res_blocks=2, device='cpu')
        worker = SelfPlayWorker(model, num_simulations=10)
        print_result("Create self-play worker", True)

        # Play one game
        print("   (Playing one game, this may take a minute...)")
        examples = worker.play_game(verbose=False)

        if len(examples) > 0:
            print_result("Generate self-play game", True, f"{len(examples)} examples")
        else:
            print_result("Generate self-play game", False, "No examples generated")
            all_passed = False

        # Check example format
        ex = examples[0]
        if ex.state.shape == (15, 11, 11):
            print_result("Training example state", True)
        else:
            print_result("Training example state", False, f"Wrong shape {ex.state.shape}")
            all_passed = False

        if ex.policy is not None and len(ex.policy) > 0:
            print_result("Training example policy", True)
        else:
            print_result("Training example policy", False)
            all_passed = False

        if -1.0 <= ex.value <= 1.0:
            print_result("Training example value", True, f"{ex.value:.1f}")
        else:
            print_result("Training example value", False, f"{ex.value}")
            all_passed = False

    except Exception as e:
        print_result("Self-play validation", False, str(e))
        all_passed = False

    return all_passed


def validate_training_pipeline() -> bool:
    """Validate training pipeline initialization"""
    print_header("6. VALIDATING TRAINING PIPELINE")

    all_passed = True

    try:
        from hnefatafl.train import Trainer, TrainingConfig

        # Create small config
        config = TrainingConfig(
            num_simulations=10,
            num_games_per_iteration=1,
            batch_size=4,
            epochs_per_iteration=1,
            num_eval_games=2,
            num_channels=32,
            num_res_blocks=2,
            device='cpu',
            verbose=False
        )
        print_result("Create training config", True)

        # Create trainer
        trainer = Trainer(config)
        print_result("Create trainer", True)

        # Check model created
        if trainer.model is not None:
            print_result("Trainer model initialized", True)
        else:
            print_result("Trainer model initialized", False)
            all_passed = False

        # Check optimizer created
        if trainer.optimizer is not None:
            print_result("Trainer optimizer initialized", True)
        else:
            print_result("Trainer optimizer initialized", False)
            all_passed = False

    except Exception as e:
        print_result("Training pipeline validation", False, str(e))
        all_passed = False

    return all_passed


def validate_gpu_availability():
    """Check GPU availability"""
    print_header("7. GPU AVAILABILITY CHECK")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print_result("CUDA available", True, f"{device_count} GPU(s) - {device_name}")

        # Test GPU tensor
        try:
            x = torch.randn(10, 10).cuda()
            print_result("GPU tensor creation", True)
        except Exception as e:
            print_result("GPU tensor creation", False, str(e))
    else:
        print_result("CUDA available", False, "No GPU found - will use CPU")
        print("        ⚠️  Training will be MUCH slower on CPU (10-50x)")


def main():
    """Run all validation tests"""
    print("\n")
    print("=" * 70)
    print("  PRE-TRAINING VALIDATION")
    print("  Run this BEFORE spending money on GPU training!")
    print("=" * 70)

    results = []

    # Run all tests
    results.append(("Imports", validate_imports()))
    results.append(("Game Engine", validate_game_engine()))
    results.append(("Neural Network", validate_neural_network()))
    results.append(("MCTS", validate_mcts()))
    results.append(("Self-Play", validate_selfplay()))
    results.append(("Training Pipeline", validate_training_pipeline()))
    validate_gpu_availability()

    # Print summary
    print_header("VALIDATION SUMMARY")

    all_passed = all([passed for _, passed in results])

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset}  {name}")

    print("\n" + "=" * 70)

    if all_passed:
        print("\n✓✓✓ ALL VALIDATION TESTS PASSED! ✓✓✓")
        print("\nYou're ready to start training!")
        print("\nNext steps:")
        print("  1. Run MVP training on CPU to verify end-to-end:")
        print("     ./venv/bin/python train_main.py --config mvp --iterations 5")
        print("  2. If MVP succeeds, rent a GPU and start real training!")
        print("=" * 70)
        return 0
    else:
        print("\n✗✗✗ SOME VALIDATION TESTS FAILED ✗✗✗")
        print("\nDO NOT start GPU training yet!")
        print("Fix the issues above first.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
