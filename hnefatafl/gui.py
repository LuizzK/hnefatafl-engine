"""
Simple Pygame GUI for Copenhagen Hnefatafl
"""

import pygame
import sys
from typing import Optional, Tuple
from hnefatafl.game import HnefataflGame, Move, Piece, Player, GameResult


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


class HnefataflGUI:
    """Pygame GUI for Hnefatafl"""

    def __init__(self, square_size: int = 60):
        pygame.init()

        self.square_size = square_size
        self.board_size = 11
        self.margin = 40

        # Window size
        self.width = self.board_size * square_size + 2 * self.margin
        self.height = self.board_size * square_size + 2 * self.margin + 100  # Extra space for info

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Copenhagen Hnefatafl")

        self.game = HnefataflGame()
        self.selected_square: Optional[Tuple[int, int]] = None
        self.legal_moves_from_selected = []

        # Fonts
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)

        self.clock = pygame.time.Clock()

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
                    x, y = self.board_to_screen(row, col)
                    center_x = x + self.square_size // 2
                    center_y = y + self.square_size // 2
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
        """Draw game information at the bottom"""
        info_y = self.margin + self.board_size * self.square_size + 20

        # Current player
        if self.game.is_game_over():
            text = self.font.render(f"Game Over: {self.game.get_result_string()}", True, RED)
        else:
            current = "Attackers (Black)" if self.game.current_player == Player.ATTACKER else "Defenders (White)"
            text = self.font.render(f"Turn: {current}", True, BLACK)
        self.screen.blit(text, (self.margin, info_y))

        # Move count
        move_text = self.small_font.render(f"Moves: {len(self.game.move_history)}", True, DARK_GRAY)
        self.screen.blit(move_text, (self.margin, info_y + 40))

        # Instructions
        if not self.game.is_game_over():
            inst = self.small_font.render("Click piece to select, click destination to move", True, DARK_GRAY)
            self.screen.blit(inst, (self.margin, info_y + 65))

    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click on the board"""
        if self.game.is_game_over():
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
                # Select this piece
                self.selected_square = (row, col)
                # Get legal moves from this square
                all_moves = self.game.get_legal_moves()
                self.legal_moves_from_selected = [
                    m for m in all_moves
                    if m.from_row == row and m.from_col == col
                ]
        else:
            # Piece already selected - try to move
            from_row, from_col = self.selected_square
            move = Move(from_row, from_col, row, col)

            if self.game.is_legal_move(move):
                # Make the move
                self.game.make_move(move)
                self.selected_square = None
                self.legal_moves_from_selected = []
            elif (row, col) == self.selected_square:
                # Clicked same square - deselect
                self.selected_square = None
                self.legal_moves_from_selected = []
            else:
                # Try to select new piece
                piece = self.game.board[row, col]
                owner = self.game.get_piece_owner(piece)

                if owner == self.game.current_player:
                    # Select new piece
                    self.selected_square = (row, col)
                    all_moves = self.game.get_legal_moves()
                    self.legal_moves_from_selected = [
                        m for m in all_moves
                        if m.from_row == row and m.from_col == col
                    ]
                else:
                    # Clicked invalid square - deselect
                    self.selected_square = None
                    self.legal_moves_from_selected = []

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
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset game
                        self.game = HnefataflGame()
                        self.selected_square = None
                        self.legal_moves_from_selected = []

            # Draw everything
            self.draw_board()
            self.draw_pieces()
            self.draw_info()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    """Launch the GUI"""
    gui = HnefataflGUI(square_size=60)
    gui.run()


if __name__ == "__main__":
    main()
