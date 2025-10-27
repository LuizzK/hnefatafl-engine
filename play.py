#!/usr/bin/env python3
"""
Play Copenhagen Hnefatafl (human vs human)
Supports both CLI and GUI modes
"""

import argparse
from hnefatafl.game import HnefataflGame, Move, Player, GameResult


def parse_move(move_str: str, game: HnefataflGame) -> Move:
    """
    Parse move string like 'f6-f8' into a Move object.

    Format: <from><to> where positions are like 'f6' (column letter + row number)
    Example: 'f6-f8' means move from f6 to f8
    """
    try:
        parts = move_str.lower().strip().split('-')
        if len(parts) != 2:
            return None

        from_pos, to_pos = parts

        # Parse from position
        from_col = ord(from_pos[0]) - ord('a')
        from_row = int(from_pos[1:]) - 1

        # Parse to position
        to_col = ord(to_pos[0]) - ord('a')
        to_row = int(to_pos[1:]) - 1

        # Validate bounds
        if not (0 <= from_row < 11 and 0 <= from_col < 11 and
                0 <= to_row < 11 and 0 <= to_col < 11):
            return None

        return Move(from_row, from_col, to_row, to_col)
    except:
        return None


def print_help():
    """Print help information"""
    print("\n=== HOW TO PLAY ===")
    print("Move format: <from>-<to>")
    print("Example: f6-f8  (move piece from f6 to f8)")
    print("\nCommands:")
    print("  help     - Show this help")
    print("  moves    - Show all legal moves")
    print("  quit     - Exit game")
    print("\nPiece symbols:")
    print("  A = Attacker")
    print("  D = Defender")
    print("  K = King")
    print("  T = Empty throne")
    print("  X = Empty corner")
    print("  . = Empty square")
    print("\nGoal:")
    print("  Defenders: Get king to any corner (X)")
    print("  Attackers: Capture the king")
    print("==================\n")


def show_legal_moves(game: HnefataflGame):
    """Show all legal moves for current player"""
    moves = game.get_legal_moves()
    print(f"\n{len(moves)} legal moves available:")

    # Group by piece
    by_piece = {}
    for move in moves:
        from_pos = f"{chr(ord('a') + move.from_col)}{move.from_row + 1}"
        if from_pos not in by_piece:
            by_piece[from_pos] = []
        to_pos = f"{chr(ord('a') + move.to_col)}{move.to_row + 1}"
        by_piece[from_pos].append(to_pos)

    for from_pos in sorted(by_piece.keys()):
        destinations = ', '.join(sorted(by_piece[from_pos]))
        print(f"  {from_pos}: {destinations}")
    print()


def main_cli():
    """Main game loop for CLI mode"""
    print("=" * 50)
    print("   COPENHAGEN HNEFATAFL")
    print("=" * 50)
    print_help()

    game = HnefataflGame()
    move_count = 0

    while not game.is_game_over():
        print(f"\n{'='*50}")
        print(f"Move {move_count + 1}")
        print(f"{'='*50}")
        print(game)
        print()

        current = "Attackers" if game.current_player == Player.ATTACKER else "Defenders"
        print(f"{current}'s turn")
        print(f"Legal moves available: {len(game.get_legal_moves())}")

        while True:
            move_input = input(f"\n{current} > ").strip().lower()

            if move_input == 'quit':
                print("Game ended by user.")
                return

            if move_input == 'help':
                print_help()
                continue

            if move_input == 'moves':
                show_legal_moves(game)
                continue

            # Try to parse as move
            move = parse_move(move_input, game)

            if move is None:
                print("Invalid format! Use: <from>-<to> (e.g., f6-f8)")
                print("Type 'help' for more information.")
                continue

            if game.is_legal_move(move):
                game.make_move(move)
                move_count += 1
                print(f"✓ Moved {move}")
                break
            else:
                print("Illegal move! That move is not allowed.")
                print("Type 'moves' to see all legal moves.")
                continue

    # Game over
    print(f"\n{'='*50}")
    print("GAME OVER!")
    print(f"{'='*50}")
    print(game)
    print()
    print(f"Result: {game.get_result_string()}")
    print(f"Total moves: {move_count}")

    winner = game.get_winner()
    if winner == Player.ATTACKER:
        print("🎉 Attackers win!")
    elif winner == Player.DEFENDER:
        print("🎉 Defenders win!")
    else:
        print("Draw")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Play Copenhagen Hnefatafl")
    parser.add_argument('--mode', choices=['cli', 'gui'], default='gui',
                        help='Interface mode: cli (command line) or gui (graphical, default)')
    parser.add_argument('--game-mode', choices=['human', 'cpu', 'setup'], default='human',
                        help='Game mode for GUI: human (human vs human), cpu (human vs cpu), setup (board setup)')

    args = parser.parse_args()

    if args.mode == 'cli':
        main_cli()
    else:
        # Launch GUI
        from hnefatafl.gui import HnefataflGUI, GameMode

        # Map argument to GameMode enum
        mode_map = {
            'human': GameMode.HUMAN_VS_HUMAN,
            'cpu': GameMode.HUMAN_VS_CPU,
            'setup': GameMode.SETUP
        }
        game_mode = mode_map[args.game_mode]

        gui = HnefataflGUI(square_size=60, mode=game_mode)
        gui.run()


if __name__ == "__main__":
    main()
