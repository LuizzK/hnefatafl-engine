"""
Enhanced Pygame GUI for Copenhagen Hnefatafl

Features:
- Human vs Human, Human vs CPU, Setup modes
- Undo/Reset buttons
- Get Hint (MCTS suggestions)
- Drag and drop for board setup
"""

import pygame
import sys
from typing import Optional, Tuple, List
from enum import Enum
from hnefatafl.game import HnefataflGame, Move, Piece, Player, GameResult
from hnefatafl.mcts import MCTS


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 90, 43)
LIGHT_BROWN = (205, 170, 125)
GREEN = (144, 238, 144)
RED = (255, 99, 71)
BLUE = (135, 206, 250)
GOLD = (255, 215, 0)
DARK_GRAY = (64, 64, 64)
LIGHT_GRAY = (200, 200, 200)
PURPLE = (147, 112, 219)


class GameMode(Enum):
    """Game mode selection"""
    HUMAN_VS_HUMAN = "Human vs Human"
    HUMAN_VS_CPU = "Human vs CPU"
    SETUP = "Setup Board"


class HnefataflGUI:
    """Enhanced Pygame GUI for Hnefatafl"""

    def __init__(self, square_size: int = 60, mode: GameMode = GameMode.HUMAN_VS_HUMAN):
        pygame.init()

        self.square_size = square_size
        self.board_size = 11
        self.margin = 40
        self.button_height = 40
        self.button_margin = 10

        # Window size
        self.width = self.board_size * square_size + 2 * self.margin
        self.height = (self.board_size * square_size + 2 * self.margin +
                      150 + self.button_height + self.button_margin)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Copenhagen Hnefatafl")

        self.game = HnefataflGame()
        self.game_history = []  # For undo
        self.selected_square: Optional[Tuple[int, int]] = None
        self.legal_moves_from_selected = []
        self.hint_move: Optional[Move] = None

        # Game mode
        self.mode = mode
        self.cpu_player = Player.DEFENDER  # CPU plays as defender by default

        # Setup mode
        self.dragging_piece: Optional[Piece] = None
        self.drag_pos: Optional[Tuple[int, int]] = None

        # MCTS for CPU and hints
        self.mcts: Optional[MCTS] = None
        self.mcts_thinking = False

        # Fonts
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        self.button_font = pygame.font.Font(None, 28)

        # Buttons
        self.buttons = {}
        self._create_buttons()

        self.clock = pygame.time.Clock()

    def _create_buttons(self):
        """Create UI buttons"""
        button_y = self.margin + self.board_size * self.square_size + 120
        button_width = 100
        button_spacing = 10

        buttons_config = [
            ("undo", "Undo", 0),
            ("reset", "Reset", 1),
            ("hint", "Hint", 2),
            ("mode", "Mode", 3),
        ]

        if self.mode == GameMode.SETUP:
            buttons_config.append(("start", "Start", 4))

        for btn_id, label, index in buttons_config:
            x = self.margin + index * (button_width + button_spacing)
            self.buttons[btn_id] = {
                'rect': pygame.Rect(x, button_y, button_width, self.button_height),
                'label': label,
                'enabled': True
            }

    def board_to_screen(self, row: int, col: int) -> Tuple[int, int]:
        """Convert board coordinates to screen coordinates"""
        x = self.margin + col * self.square_size
        y = self.margin + (self.board_size - 1 - row) * self.square_size
        return x, y

    def screen_to_board(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Convert screen coordinates to board coordinates"""
        if x < self.margin or y < self.margin:
            return None

        col = (x - self.margin) // self.square_size
        row = self.board_size - 1 - (y - self.margin) // self.square_size

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return (row, col)
        return None

    def draw_board(self):
        """Draw the game board"""
        self.screen.fill(LIGHT_GRAY)

        # Draw squares
        for row in range(self.board_size):
            for col in range(self.board_size):
                x, y = self.board_to_screen(row, col)

                # Determine square color
                if (row, col) == self.game.THRONE:
                    color = GOLD if self.game.board[row, col] == Piece.EMPTY else BROWN
                elif (row, col) in self.game.CORNERS:
                    color = RED if self.game.board[row, col] == Piece.EMPTY else BROWN
                elif (row + col) % 2 == 0:
                    color = LIGHT_BROWN
                else:
                    color = BROWN

                # Highlight selected square
                if self.selected_square and (row, col) == self.selected_square:
                    color = GREEN

                # Highlight legal move destinations
                if self.selected_square:
                    for move in self.legal_moves_from_selected:
                        if move.to_row == row and move.to_col == col:
                            color = BLUE

                # Highlight hint move
                if self.hint_move:
                    if ((self.hint_move.from_row, self.hint_move.from_col) == (row, col) or
                        (self.hint_move.to_row, self.hint_move.to_col) == (row, col)):
                        color = PURPLE

                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
                pygame.draw.rect(self.screen, BLACK, (x, y, self.square_size, self.square_size), 1)

        # Draw coordinate labels
        for i in range(self.board_size):
            # Column labels (a-k) - bottom only
            label = self.small_font.render(chr(ord('a') + i), True, BLACK)
            x, y = self.board_to_screen(0, i)
            self.screen.blit(label, (x + self.square_size // 2 - 5, y + self.square_size + 5))

            # Row labels (1-11) - both sides
            label = self.small_font.render(str(i + 1), True, BLACK)
            x, y = self.board_to_screen(i, 0)
            self.screen.blit(label, (x - 30, y + self.square_size // 2 - 10))
            x, y = self.board_to_screen(i, self.board_size - 1)
            self.screen.blit(label, (x + self.square_size + 10, y + self.square_size // 2 - 10))

    def draw_pieces(self):
        """Draw all pieces on the board"""
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board[row, col]
                if piece != Piece.EMPTY:
                    # Skip if we're dragging this piece
                    if (self.dragging_piece and self.selected_square and
                        (row, col) == self.selected_square):
                        continue

                    x, y = self.board_to_screen(row, col)
                    self._draw_piece(piece, x + self.square_size // 2, y + self.square_size // 2)

        # Draw dragging piece at mouse position
        if self.dragging_piece and self.drag_pos:
            self._draw_piece(self.dragging_piece, self.drag_pos[0], self.drag_pos[1])

    def _draw_piece(self, piece: Piece, center_x: int, center_y: int):
        """Draw a single piece at given position"""
        radius = self.square_size // 3

        if piece == Piece.ATTACKER:
            # Black circle
            pygame.draw.circle(self.screen, BLACK, (center_x, center_y), radius)
            pygame.draw.circle(self.screen, WHITE, (center_x, center_y), radius, 2)
        elif piece == Piece.DEFENDER:
            # White circle
            pygame.draw.circle(self.screen, WHITE, (center_x, center_y), radius)
            pygame.draw.circle(self.screen, BLACK, (center_x, center_y), radius, 2)
        elif piece == Piece.KING:
            # White circle with crown
            pygame.draw.circle(self.screen, WHITE, (center_x, center_y), radius)
            pygame.draw.circle(self.screen, GOLD, (center_x, center_y), radius, 3)
            # Draw crown symbol
            crown_points = [
                (center_x, center_y - radius // 2),
                (center_x - radius // 3, center_y - radius // 4),
                (center_x, center_y),
                (center_x + radius // 3, center_y - radius // 4),
                (center_x, center_y - radius // 2)
            ]
            pygame.draw.lines(self.screen, GOLD, False, crown_points, 3)

    def draw_info(self):
        """Draw game information"""
        info_y = self.margin + self.board_size * self.square_size + 20

        # Game mode
        mode_text = f"Mode: {self.mode.value}"
        if self.mode == GameMode.HUMAN_VS_CPU:
            cpu_side = "Defender" if self.cpu_player == Player.DEFENDER else "Attacker"
            mode_text += f" (CPU: {cpu_side})"
        text = self.small_font.render(mode_text, True, BLACK)
        self.screen.blit(text, (self.margin, info_y))

        # Current player / Game state
        if self.game.is_game_over():
            text = self.font.render(f"Game Over: {self.game.get_result_string()}", True, RED)
        elif self.mcts_thinking:
            text = self.font.render("CPU thinking...", True, BLUE)
        elif self.mode == GameMode.SETUP:
            text = self.font.render("Setup Mode - Drag pieces to place", True, BLACK)
        else:
            current = "Attackers (Black)" if self.game.current_player == Player.ATTACKER else "Defenders (White)"
            text = self.font.render(f"Turn: {current}", True, BLACK)
        self.screen.blit(text, (self.margin, info_y + 25))

        # Move count
        move_text = self.small_font.render(f"Moves: {len(self.game_history)}", True, DARK_GRAY)
        self.screen.blit(move_text, (self.margin, info_y + 60))

        # Hint display
        if self.hint_move:
            hint_text = self.small_font.render(f"Hint: {self.hint_move}", True, PURPLE)
            self.screen.blit(hint_text, (self.margin + 150, info_y + 60))

    def draw_buttons(self):
        """Draw UI buttons"""
        for btn_id, btn in self.buttons.items():
            rect = btn['rect']
            label = btn['label']
            enabled = btn['enabled']

            # Button color
            color = BLUE if enabled else DARK_GRAY
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 2)

            # Button text
            text = self.button_font.render(label, True, WHITE if enabled else LIGHT_GRAY)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click"""
        # Check buttons first
        for btn_id, btn in self.buttons.items():
            if btn['rect'].collidepoint(pos) and btn['enabled']:
                self._handle_button(btn_id)
                return

        # Handle board click based on mode
        if self.mode == GameMode.SETUP:
            self._handle_setup_click(pos)
        else:
            self._handle_game_click(pos)

    def _handle_button(self, btn_id: str):
        """Handle button clicks"""
        if btn_id == "undo":
            self.undo_move()
        elif btn_id == "reset":
            self.reset_game()
        elif btn_id == "hint":
            self.get_hint()
        elif btn_id == "mode":
            self.cycle_mode()
        elif btn_id == "start":
            self.start_game_from_setup()

    def undo_move(self):
        """Undo last move"""
        if len(self.game_history) > 0:
            self.game = self.game_history.pop().copy()
            self.hint_move = None
            self.selected_square = None
            self.legal_moves_from_selected = []

    def reset_game(self):
        """Reset game to initial position"""
        self.game = HnefataflGame()
        self.game_history = []
        self.hint_move = None
        self.selected_square = None
        self.legal_moves_from_selected = []

    def cycle_mode(self):
        """Cycle through game modes"""
        modes = [GameMode.HUMAN_VS_HUMAN, GameMode.HUMAN_VS_CPU, GameMode.SETUP]
        current_idx = modes.index(self.mode)
        self.mode = modes[(current_idx + 1) % len(modes)]
        self._create_buttons()  # Recreate buttons for new mode
        self.reset_game()

    def start_game_from_setup(self):
        """Start game from setup mode"""
        self.mode = GameMode.HUMAN_VS_HUMAN
        self.game_history = []
        self._create_buttons()

    def get_hint(self):
        """Get MCTS hint for current position"""
        if self.game.is_game_over() or self.mcts_thinking:
            return

        # Create MCTS if needed
        if self.mcts is None:
            self.mcts = MCTS(neural_network=None, num_simulations=100)

        # Get best move (this will be slow - should be done in background)
        self.mcts_thinking = True
        pygame.display.flip()

        move, _ = self.mcts.search(self.game.copy())
        self.hint_move = move
        self.mcts_thinking = False

    def _handle_setup_click(self, pos: Tuple[int, int]):
        """Handle click in setup mode"""
        board_pos = self.screen_to_board(pos[0], pos[1])
        if not board_pos:
            return

        row, col = board_pos
        piece = self.game.board[row, col]

        if piece != Piece.EMPTY:
            # Start dragging this piece
            self.selected_square = (row, col)
            self.dragging_piece = piece
            self.drag_pos = pos
        else:
            # Place piece or remove dragging piece
            if self.dragging_piece:
                self.game.board[row, col] = self.dragging_piece
                if self.selected_square:
                    old_row, old_col = self.selected_square
                    self.game.board[old_row, old_col] = Piece.EMPTY
                self.dragging_piece = None
                self.selected_square = None

    def _handle_game_click(self, pos: Tuple[int, int]):
        """Handle click during game"""
        if self.game.is_game_over() or self.mcts_thinking:
            return

        # Don't allow moves if it's CPU's turn
        if self.mode == GameMode.HUMAN_VS_CPU and self.game.current_player == self.cpu_player:
            return

        board_pos = self.screen_to_board(pos[0], pos[1])
        if not board_pos:
            return

        row, col = board_pos

        # If no piece selected, try to select one
        if not self.selected_square:
            piece = self.game.board[row, col]
            owner = self.game.get_piece_owner(piece)

            if owner == self.game.current_player:
                self.selected_square = (row, col)
                all_moves = self.game.get_legal_moves()
                self.legal_moves_from_selected = [
                    m for m in all_moves
                    if m.from_row == row and m.from_col == col
                ]
                self.hint_move = None  # Clear hint
        else:
            # Try to move
            from_row, from_col = self.selected_square
            move = Move(from_row, from_col, row, col)

            if self.game.is_legal_move(move):
                # Save for undo
                self.game_history.append(self.game.copy())
                # Make move
                self.game.make_move(move)
                self.selected_square = None
                self.legal_moves_from_selected = []
                self.hint_move = None
            elif (row, col) == self.selected_square:
                # Deselect
                self.selected_square = None
                self.legal_moves_from_selected = []
            else:
                # Try to select new piece
                piece = self.game.board[row, col]
                owner = self.game.get_piece_owner(piece)

                if owner == self.game.current_player:
                    self.selected_square = (row, col)
                    all_moves = self.game.get_legal_moves()
                    self.legal_moves_from_selected = [
                        m for m in all_moves
                        if m.from_row == row and m.from_col == col
                    ]

    def handle_drag(self, pos: Tuple[int, int]):
        """Handle mouse drag for setup mode"""
        if self.mode == GameMode.SETUP and self.dragging_piece:
            self.drag_pos = pos

    def handle_release(self, pos: Tuple[int, int]):
        """Handle mouse release for setup mode"""
        if self.mode == GameMode.SETUP and self.dragging_piece:
            board_pos = self.screen_to_board(pos[0], pos[1])
            if board_pos:
                row, col = board_pos
                # Place piece
                self.game.board[row, col] = self.dragging_piece
                if self.selected_square:
                    old_row, old_col = self.selected_square
                    self.game.board[old_row, old_col] = Piece.EMPTY

            self.dragging_piece = None
            self.selected_square = None
            self.drag_pos = None

    def make_cpu_move(self):
        """Make CPU move if it's CPU's turn"""
        if (self.mode == GameMode.HUMAN_VS_CPU and
            not self.game.is_game_over() and
            self.game.current_player == self.cpu_player and
            not self.mcts_thinking):

            # Create MCTS if needed
            if self.mcts is None:
                self.mcts = MCTS(neural_network=None, num_simulations=200)

            self.mcts_thinking = True
            pygame.display.flip()

            # Get best move
            move, _ = self.mcts.search(self.game.copy())

            # Save for undo
            self.game_history.append(self.game.copy())
            # Make move
            self.game.make_move(move)
            self.mcts_thinking = False

    def run(self):
        """Main game loop"""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_drag(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.handle_release(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_u:
                        self.undo_move()
                    elif event.key == pygame.K_h:
                        self.get_hint()

            # Make CPU move if needed
            if not self.mcts_thinking:
                self.make_cpu_move()

            # Draw everything
            self.draw_board()
            self.draw_pieces()
            self.draw_info()
            self.draw_buttons()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    """Launch the GUI"""
    gui = HnefataflGUI(square_size=60, mode=GameMode.HUMAN_VS_HUMAN)
    gui.run()


if __name__ == "__main__":
    main()
