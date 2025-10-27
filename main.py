"""
Main entry point for Copenhagen Hnefatafl Engine
"""

import argparse
from hnefatafl.game import HnefataflGame


def main():
    parser = argparse.ArgumentParser(description="Copenhagen Hnefatafl Engine")
    parser.add_argument('--mode', choices=['play', 'train', 'gui'], default='gui',
                        help='Mode: play (CLI), train (self-play training), or gui (visual interface)')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')

    args = parser.parse_args()

    if args.mode == 'play':
        # CLI play mode
        game = HnefataflGame()
        print("Copenhagen Hnefatafl - CLI Mode")
        print("Starting position:")
        print(game)

    elif args.mode == 'train':
        # Training mode
        print("Training mode - To be implemented")

    elif args.mode == 'gui':
        # GUI mode
        print("GUI mode - To be implemented")


if __name__ == "__main__":
    main()
