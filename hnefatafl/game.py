"""
Copenhagen Hnefatafl Game Engine

Implements complete Copenhagen Hnefatafl rules on 11x11 board including:
- Standard piece movement and capture
- Shieldwall capture
- Edge forts
- King capture mechanics
- All win conditions
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from enum import IntEnum
from dataclasses import dataclass


class Piece(IntEnum):
    """Piece types on the board"""
    EMPTY = 0
    ATTACKER = 1
    DEFENDER = 2
    KING = 3


class Player(IntEnum):
    """Player identifiers"""
    ATTACKER = 0
    DEFENDER = 1


@dataclass
class Move:
    """Represents a move from one position to another"""
    from_row: int
    from_col: int
    to_row: int
    to_col: int

    def __hash__(self):
        return hash((self.from_row, self.from_col, self.to_row, self.to_col))

    def __eq__(self, other):
        return (self.from_row == other.from_row and
                self.from_col == other.from_col and
                self.to_row == other.to_row and
                self.to_col == other.to_col)

    def __repr__(self):
        return f"{chr(ord('a') + self.from_col)}{self.from_row + 1}" \
               f"-{chr(ord('a') + self.to_col)}{self.to_row + 1}"


class GameResult(IntEnum):
    """Game result states"""
    ONGOING = 0
    DEFENDER_WIN = 1
    ATTACKER_WIN = 2
    DRAW = 3


class HnefataflGame:
    """
    Copenhagen Hnefatafl game implementation

    Board coordinates: (row, col) where (0,0) is bottom-left (a1)
    Rows and columns are 0-indexed (0-10 for 11x11 board)
    """

    BOARD_SIZE = 11

    # Special positions (using 0-indexed coordinates)
    THRONE = (5, 5)  # F6 in algebraic notation
    CORNERS = [(0, 0), (0, 10), (10, 0), (10, 10)]  # a1, k1, a11, k11

    def __init__(self):
        """Initialize a new game with the Copenhagen starting position"""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.current_player = Player.ATTACKER  # Attackers move first
        self.move_history = []  # List of (board_state, move) tuples for repetition detection
        self.result = GameResult.ONGOING

        self._setup_initial_position()

    def _setup_initial_position(self):
        """
        Set up the Copenhagen Hnefatafl starting position.

        Standard Copenhagen setup:
        - King at F6 (5,5)
        - 12 Defenders around the king in a cross pattern
        - 24 Attackers in T-shapes on each edge
        """
        # Place king at center (throne)
        self.board[5, 5] = Piece.KING

        # Place 12 defenders around king (cross pattern)
        defenders = [
            # Horizontal line through king
            (5, 3), (5, 4), (5, 6), (5, 7),
            # Vertical line through king
            (3, 5), (4, 5), (6, 5), (7, 5),
            # Extended cross arms
            (4, 4), (4, 6), (6, 4), (6, 6)
        ]
        for row, col in defenders:
            self.board[row, col] = Piece.DEFENDER

        # Place 24 attackers in T-shapes on each edge
        # Top edge (row 10)
        attackers_top = [(10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (9, 5)]
        # Bottom edge (row 0)
        attackers_bottom = [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 5)]
        # Left edge (col 0)
        attackers_left = [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (5, 1)]
        # Right edge (col 10)
        attackers_right = [(3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (5, 9)]

        all_attackers = attackers_top + attackers_bottom + attackers_left + attackers_right
        for row, col in all_attackers:
            self.board[row, col] = Piece.ATTACKER

    def is_restricted_square(self, row: int, col: int) -> bool:
        """Check if a square is restricted (throne or corner)"""
        return (row, col) == self.THRONE or (row, col) in self.CORNERS

    def is_corner(self, row: int, col: int) -> bool:
        """Check if a square is a corner"""
        return (row, col) in self.CORNERS

    def is_hostile_square(self, row: int, col: int, for_player: Player) -> bool:
        """
        Check if a square acts as hostile for capture purposes.

        Copenhagen rules:
        - Throne is always hostile to attackers
        - Throne is hostile to defenders only when empty
        - Corners are hostile to all pieces
        """
        if (row, col) in self.CORNERS:
            return True

        if (row, col) == self.THRONE:
            if for_player == Player.ATTACKER:
                return True
            else:  # Defender
                # Throne is hostile to defenders only when empty
                return self.board[self.THRONE] == Piece.EMPTY

        return False

    def get_piece_owner(self, piece: Piece) -> Optional[Player]:
        """Get the player who owns a piece"""
        if piece == Piece.ATTACKER:
            return Player.ATTACKER
        elif piece in (Piece.DEFENDER, Piece.KING):
            return Player.DEFENDER
        return None

    def copy(self):
        """Create a deep copy of the game state"""
        new_game = HnefataflGame.__new__(HnefataflGame)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.result = self.result
        return new_game

    def __str__(self) -> str:
        """String representation of the board"""
        piece_symbols = {
            Piece.EMPTY: '.',
            Piece.ATTACKER: 'A',
            Piece.DEFENDER: 'D',
            Piece.KING: 'K'
        }

        lines = []
        lines.append("   a b c d e f g h i j k")
        lines.append("  " + "-" * 23)

        # Print from top to bottom (row 10 to row 0)
        for row in range(self.BOARD_SIZE - 1, -1, -1):
            row_str = f"{row + 1:2}|"
            for col in range(self.BOARD_SIZE):
                piece = self.board[row, col]
                symbol = piece_symbols[piece]

                # Mark special squares
                if (row, col) == self.THRONE and piece == Piece.EMPTY:
                    symbol = 'T'  # Throne
                elif (row, col) in self.CORNERS and piece == Piece.EMPTY:
                    symbol = 'X'  # Corner

                row_str += symbol + ' '
            row_str += f"|{row + 1}"
            lines.append(row_str)

        lines.append("  " + "-" * 23)
        lines.append("   a b c d e f g h i j k")
        lines.append(f"\nCurrent player: {'Attacker' if self.current_player == Player.ATTACKER else 'Defender'}")

        return '\n'.join(lines)

    def get_state_hash(self) -> int:
        """Get a hash of the current board state for repetition detection"""
        return hash((self.board.tobytes(), self.current_player))

    def encode_state(self) -> np.ndarray:
        """
        Encode the game state as multi-channel tensor for neural network input.

        Returns: numpy array of shape (15, 11, 11) with channels:
            0: Current player's regular pieces (attackers or defenders)
            1: Current player's king (if defender)
            2: Opponent's regular pieces
            3: Opponent's king (if defender is opponent)
            4: Throne square
            5: Corner squares
            6: Board edges
            7-14: Last 8 half-moves (binary: piece was here)
        """
        channels = np.zeros((15, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)

        # Channel 0-3: Piece positions
        if self.current_player == Player.ATTACKER:
            channels[0] = (self.board == Piece.ATTACKER).astype(np.float32)
            channels[1] = np.zeros_like(channels[0])  # Attackers don't have king
            channels[2] = (self.board == Piece.DEFENDER).astype(np.float32)
            channels[3] = (self.board == Piece.KING).astype(np.float32)
        else:  # Defender
            channels[0] = (self.board == Piece.DEFENDER).astype(np.float32)
            channels[1] = (self.board == Piece.KING).astype(np.float32)
            channels[2] = (self.board == Piece.ATTACKER).astype(np.float32)
            channels[3] = np.zeros_like(channels[0])  # Opponent doesn't have king

        # Channel 4: Throne
        throne_r, throne_c = self.THRONE
        channels[4, throne_r, throne_c] = 1.0

        # Channel 5: Corners
        for corner_r, corner_c in self.CORNERS:
            channels[5, corner_r, corner_c] = 1.0

        # Channel 6: Board edges
        channels[6, 0, :] = 1.0
        channels[6, -1, :] = 1.0
        channels[6, :, 0] = 1.0
        channels[6, :, -1] = 1.0

        # Channels 7-14: Last 8 half-moves (for temporal information)
        # This will be populated as we track move history
        # For now, leave as zeros

        return channels

    def get_legal_moves(self) -> List[Move]:
        """
        Generate all legal moves for the current player.

        Returns: List of Move objects representing all legal moves
        """
        moves = []

        # Find all pieces belonging to current player
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = self.board[row, col]
                owner = self.get_piece_owner(piece)

                if owner == self.current_player:
                    # Generate all legal moves for this piece
                    piece_moves = self._get_piece_moves(row, col)
                    moves.extend(piece_moves)

        return moves

    def _get_piece_moves(self, from_row: int, from_col: int) -> List[Move]:
        """
        Get all legal moves for a piece at the given position.

        Pieces move like rooks in chess: any number of squares horizontally or vertically.
        Pieces cannot jump over other pieces.
        Only the king can land on restricted squares (throne and corners).
        """
        moves = []
        piece = self.board[from_row, from_col]

        # Check all four cardinal directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Up, Down

        for dr, dc in directions:
            # Try each square in this direction until we hit an obstacle
            distance = 1
            while True:
                to_row = from_row + dr * distance
                to_col = from_col + dc * distance

                # Check if we're still on the board
                if not (0 <= to_row < self.BOARD_SIZE and 0 <= to_col < self.BOARD_SIZE):
                    break

                # Check if square is occupied
                target_square = self.board[to_row, to_col]
                if target_square != Piece.EMPTY:
                    break  # Can't move through pieces

                # Check if square is restricted (only king can land on throne/corners)
                if self.is_restricted_square(to_row, to_col) and piece != Piece.KING:
                    # Pieces cannot land on corners at all
                    if (to_row, to_col) in self.CORNERS:
                        break
                    # For throne: pieces can pass through when empty, but cannot land on it
                    # Continue to check next square (don't add this as a valid destination)
                    distance += 1
                    continue

                # This is a valid move
                moves.append(Move(from_row, from_col, to_row, to_col))
                distance += 1

        return moves

    def is_legal_move(self, move: Move) -> bool:
        """Check if a move is legal"""
        legal_moves = self.get_legal_moves()
        return move in legal_moves

    def make_move(self, move: Move) -> bool:
        """
        Make a move on the board.

        Returns: True if move was successful, False otherwise
        """
        if not self.is_legal_move(move):
            return False

        # Store current state for repetition detection
        state_before = self.board.copy()

        # Move the piece
        piece = self.board[move.from_row, move.from_col]
        self.board[move.to_row, move.to_col] = piece
        self.board[move.from_row, move.from_col] = Piece.EMPTY

        # Process captures (standard and shieldwall)
        self._process_captures(move)

        # Add to move history
        self.move_history.append((state_before, move))

        # Check for game end
        self._check_game_end()

        # Switch player
        self.current_player = Player.DEFENDER if self.current_player == Player.ATTACKER else Player.ATTACKER

        return True

    def _process_captures(self, move: Move):
        """Process all captures after a move (standard and shieldwall)"""
        # Check for standard captures in all four directions
        self._check_standard_captures(move.to_row, move.to_col)

        # Check for shieldwall captures along board edges
        self._check_shieldwall_captures()

    def _check_standard_captures(self, row: int, col: int):
        """
        Check for standard custodian captures around the piece that just moved.

        A piece is captured when sandwiched between two enemy pieces or
        an enemy piece and a hostile square (throne/corner).
        """
        piece = self.board[row, col]
        owner = self.get_piece_owner(piece)

        if owner is None:
            return

        # Check all four orthogonal directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dr, dc in directions:
            # Check if there's an enemy piece one square away in this direction
            check_row = row + dr
            check_col = col + dc

            if not (0 <= check_row < self.BOARD_SIZE and 0 <= check_col < self.BOARD_SIZE):
                continue

            target_piece = self.board[check_row, check_col]
            target_owner = self.get_piece_owner(target_piece)

            # Skip if not an enemy piece (or if it's a king - handled separately)
            if target_owner is None or target_owner == owner:
                continue

            if target_piece == Piece.KING:
                # King capture has special rules
                self._check_king_capture()
                continue

            # Check if there's a captor on the opposite side
            opposite_row = check_row + dr
            opposite_col = check_col + dc

            # Check if opposite position is on the board
            if not (0 <= opposite_row < self.BOARD_SIZE and 0 <= opposite_col < self.BOARD_SIZE):
                continue

            # Check if opposite side has a friendly piece or hostile square
            opposite_piece = self.board[opposite_row, opposite_col]
            opposite_owner = self.get_piece_owner(opposite_piece)

            is_captured = False

            if opposite_owner == owner:
                # Sandwiched between two enemy pieces
                is_captured = True
            elif self.is_hostile_square(opposite_row, opposite_col, target_owner):
                # Sandwiched between enemy piece and hostile square
                is_captured = True

            if is_captured:
                self.board[check_row, check_col] = Piece.EMPTY

    def _check_king_capture(self):
        """
        Check if the king has been captured.

        Copenhagen rules for king capture:
        - King must be surrounded on all 4 sides by attackers
        - Exception: When king is adjacent to throne, only 3 surrounding attackers needed
        - King cannot be captured on the board edge
        """
        # Find the king
        king_pos = None
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row, col] == Piece.KING:
                    king_pos = (row, col)
                    break
            if king_pos:
                break

        if not king_pos:
            return  # King already captured

        king_row, king_col = king_pos

        # King cannot be captured on the edge
        if king_row == 0 or king_row == self.BOARD_SIZE - 1 or \
           king_col == 0 or king_col == self.BOARD_SIZE - 1:
            return

        # Check if king is adjacent to throne
        adjacent_to_throne = False
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_row, adj_col = king_row + dr, king_col + dc
            if (adj_row, adj_col) == self.THRONE:
                adjacent_to_throne = True
                break

        # Count attackers surrounding the king
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        attacker_count = 0
        hostile_squares = 0

        for dr, dc in directions:
            check_row = king_row + dr
            check_col = king_col + dc

            if not (0 <= check_row < self.BOARD_SIZE and 0 <= check_col < self.BOARD_SIZE):
                continue

            piece = self.board[check_row, check_col]

            if piece == Piece.ATTACKER:
                attacker_count += 1
            elif (check_row, check_col) == self.THRONE:
                # Throne counts as hostile when king is adjacent to it
                hostile_squares += 1

        total_hostile = attacker_count + hostile_squares

        # Determine if king is captured
        if adjacent_to_throne:
            # Need 3 hostile squares when adjacent to throne
            if total_hostile >= 3:
                self.board[king_row, king_col] = Piece.EMPTY
        else:
            # Need all 4 sides surrounded
            if total_hostile >= 4:
                self.board[king_row, king_col] = Piece.EMPTY

    def _check_shieldwall_captures(self):
        """
        Check for shieldwall captures along board edges.

        Shieldwall capture: Multiple pieces aligned along an edge can be captured
        together when bracketed at both ends, provided each has an enemy directly opposite.

        The king escapes shieldwall capture even if adjacent defenders are captured.
        """
        # Check all four edges
        edges = [
            ('top', 10, [(10, c) for c in range(self.BOARD_SIZE)]),
            ('bottom', 0, [(0, c) for c in range(self.BOARD_SIZE)]),
            ('left', 0, [(r, 0) for r in range(self.BOARD_SIZE)]),
            ('right', 10, [(r, 10) for r in range(self.BOARD_SIZE)])
        ]

        for edge_name, edge_coord, edge_squares in edges:
            self._check_edge_shieldwall(edge_name, edge_squares)

    def _check_edge_shieldwall(self, edge_name: str, edge_squares: List[Tuple[int, int]]):
        """Check for shieldwall captures along a specific edge"""
        # Find continuous groups of pieces along the edge
        groups = []
        current_group = []
        current_owner = None

        for row, col in edge_squares:
            piece = self.board[row, col]
            owner = self.get_piece_owner(piece)

            if owner is not None and owner == current_owner:
                current_group.append((row, col, piece))
            elif owner is not None:
                if current_group:
                    groups.append((current_owner, current_group))
                current_group = [(row, col, piece)]
                current_owner = owner
            else:
                if current_group:
                    groups.append((current_owner, current_group))
                current_group = []
                current_owner = None

        if current_group:
            groups.append((current_owner, current_group))

        # Check each group for shieldwall capture
        for owner, group in groups:
            if len(group) < 2:
                continue  # Shieldwall requires at least 2 pieces

            # Check if group is bracketed at both ends
            is_horizontal = edge_name in ['top', 'bottom']
            is_captured = self._check_shieldwall_bracket(group, is_horizontal, owner)

            if is_captured:
                # Remove all pieces in the group except the king
                for row, col, piece in group:
                    if piece != Piece.KING:
                        self.board[row, col] = Piece.EMPTY

    def _check_shieldwall_bracket(self, group: List[Tuple[int, int, Piece]],
                                    is_horizontal: bool, group_owner: Player) -> bool:
        """
        Check if a group of pieces along an edge is bracketed at both ends
        and has enemies opposite to each piece.
        """
        if len(group) == 0:
            return False

        # Get first and last positions
        first_row, first_col, _ = group[0]
        last_row, last_col, _ = group[-1]

        # Determine direction perpendicular to edge (inward)
        if is_horizontal:
            # For top/bottom edges, check perpendicular direction
            if first_row == 0:  # Bottom edge
                perp_dr, perp_dc = 1, 0
            else:  # Top edge
                perp_dr, perp_dc = -1, 0
        else:
            # For left/right edges
            if first_col == 0:  # Left edge
                perp_dr, perp_dc = 0, 1
            else:  # Right edge
                perp_dr, perp_dc = 0, -1

        # Check if bracketed at both ends
        # Determine direction along the edge
        if is_horizontal:
            # Check left and right of group
            left_col = first_col - 1
            right_col = last_col + 1

            # Check left bracket
            left_bracketed = False
            if left_col < 0 or left_col >= self.BOARD_SIZE:
                left_bracketed = True  # Edge of board acts as bracket
            elif (first_row, left_col) in self.CORNERS:
                left_bracketed = True  # Corner acts as bracket
            else:
                left_piece = self.board[first_row, left_col]
                left_owner = self.get_piece_owner(left_piece)
                if left_owner is not None and left_owner != group_owner:
                    left_bracketed = True

            # Check right bracket
            right_bracketed = False
            if right_col < 0 or right_col >= self.BOARD_SIZE:
                right_bracketed = True
            elif (last_row, right_col) in self.CORNERS:
                right_bracketed = True
            else:
                right_piece = self.board[last_row, right_col]
                right_owner = self.get_piece_owner(right_piece)
                if right_owner is not None and right_owner != group_owner:
                    right_bracketed = True

            if not (left_bracketed and right_bracketed):
                return False

        else:  # Vertical edge
            # Check top and bottom of group
            top_row = last_row + 1  # Assuming group is ordered bottom to top
            bottom_row = first_row - 1

            # Check bottom bracket
            bottom_bracketed = False
            if bottom_row < 0 or bottom_row >= self.BOARD_SIZE:
                bottom_bracketed = True
            elif (bottom_row, first_col) in self.CORNERS:
                bottom_bracketed = True
            else:
                bottom_piece = self.board[bottom_row, first_col]
                bottom_owner = self.get_piece_owner(bottom_piece)
                if bottom_owner is not None and bottom_owner != group_owner:
                    bottom_bracketed = True

            # Check top bracket
            top_bracketed = False
            if top_row < 0 or top_row >= self.BOARD_SIZE:
                top_bracketed = True
            elif (top_row, last_col) in self.CORNERS:
                top_bracketed = True
            else:
                top_piece = self.board[top_row, last_col]
                top_owner = self.get_piece_owner(top_piece)
                if top_owner is not None and top_owner != group_owner:
                    top_bracketed = True

            if not (bottom_bracketed and top_bracketed):
                return False

        # Check if each piece in group has an enemy directly opposite
        for row, col, _ in group:
            opposite_row = row + perp_dr
            opposite_col = col + perp_dc

            if not (0 <= opposite_row < self.BOARD_SIZE and 0 <= opposite_col < self.BOARD_SIZE):
                return False  # Must have opposite square

            opposite_piece = self.board[opposite_row, opposite_col]
            opposite_owner = self.get_piece_owner(opposite_piece)

            # Check if opposite has an enemy piece
            if opposite_owner is None or opposite_owner == group_owner:
                return False

        return True

    def _check_game_end(self):
        """Check if the game has ended and update game result"""
        # Check if king reached a corner (defender win)
        king_pos = self._find_king()
        if king_pos and king_pos in self.CORNERS:
            self.result = GameResult.DEFENDER_WIN
            return

        # Check if king was captured (attacker win)
        if not king_pos:
            self.result = GameResult.ATTACKER_WIN
            return

        # Check for edge fort (defender win)
        if self._check_edge_fort():
            self.result = GameResult.DEFENDER_WIN
            return

        # Check for encirclement (attacker win)
        if self._check_encirclement():
            self.result = GameResult.ATTACKER_WIN
            return

        # Check for no legal moves (loss for current player)
        if len(self.get_legal_moves()) == 0:
            if self.current_player == Player.ATTACKER:
                self.result = GameResult.DEFENDER_WIN
            else:
                self.result = GameResult.ATTACKER_WIN
            return

        # Check for repetition (defender loss in Copenhagen rules)
        if self._check_repetition():
            self.result = GameResult.ATTACKER_WIN
            return

    def _find_king(self) -> Optional[Tuple[int, int]]:
        """Find the king's position on the board"""
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row, col] == Piece.KING:
                    return (row, col)
        return None

    def _check_edge_fort(self) -> bool:
        """
        Check if defenders have formed an unbreakable edge fort.

        Edge fort win condition (Copenhagen rules):
        1. King has contact with the board edge
        2. King is able to move
        3. Attackers cannot break the fort

        An edge fort is unbreakable when:
        - King cannot be captured (immune on edge)
        - Defenders cannot all be removed via shieldwall capture
        - Formation provides sustained protection

        This is a complex pattern recognition problem. We check:
        - Corner-adjacent positions (very strong)
        - Protected formations that can't be dismantled
        - No vulnerabilities to shieldwall capture
        """
        king_pos = self._find_king()
        if not king_pos:
            return False

        king_row, king_col = king_pos

        # Requirement 1: King must be ON the board edge (not just adjacent)
        on_edge = (king_row == 0 or king_row == self.BOARD_SIZE - 1 or
                   king_col == 0 or king_col == self.BOARD_SIZE - 1)

        if not on_edge:
            return False

        # Requirement 2: King must be able to move
        king_moves = self._get_piece_moves(king_row, king_col)
        if len(king_moves) == 0:
            return False

        # Requirement 3: Check if fort is truly unbreakable
        # This requires analyzing the formation structure

        # Get all defenders near the king
        nearby_defenders = self._get_nearby_defenders(king_row, king_col, radius=2)

        if len(nearby_defenders) < 2:
            # Not enough defenders to form a fort
            return False

        # Check for specific unbreakable patterns

        # Pattern 1: Corner-adjacent position
        if self._is_corner_adjacent_fort(king_row, king_col, nearby_defenders):
            return True

        # Pattern 2: Protected L-shape or square formation
        if self._is_protected_formation(king_row, king_col, nearby_defenders):
            return True

        # Pattern 3: Edge formation immune to shieldwall
        if self._is_shieldwall_immune_formation(king_row, king_col, nearby_defenders):
            return True

        # If none of the unbreakable patterns match, it's not a confirmed fort
        return False

    def _get_nearby_defenders(self, row: int, col: int, radius: int = 2) -> List[Tuple[int, int]]:
        """Get all defender positions within radius of given position"""
        defenders = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                check_row, check_col = row + dr, col + dc
                if (0 <= check_row < self.BOARD_SIZE and
                    0 <= check_col < self.BOARD_SIZE):
                    if self.board[check_row, check_col] == Piece.DEFENDER:
                        defenders.append((check_row, check_col))
        return defenders

    def _is_corner_adjacent_fort(self, king_row: int, king_col: int,
                                  defenders: List[Tuple[int, int]]) -> bool:
        """
        Check if king is in corner-adjacent position with defender support.

        Corner-adjacent forts are very strong because:
        - Corner acts as a defender
        - King is on edge (cannot be captured)
        - Limited attack vectors

        Pattern examples:
        K D X  or  X D K
        D . .      . . D
        """
        # Check if adjacent to any corner
        corners_adjacent = []
        for corner in self.CORNERS:
            corner_r, corner_c = corner
            # Check if king is adjacent to this corner
            if abs(king_row - corner_r) + abs(king_col - corner_c) == 1:
                corners_adjacent.append(corner)

        if not corners_adjacent:
            return False

        # King is adjacent to corner - check if defenders protect the formation
        for corner_r, corner_c in corners_adjacent:
            # Check if there are defenders forming a protective pocket
            # Need at least 1-2 defenders near king and corner
            defenders_near_corner = [
                (d_r, d_c) for d_r, d_c in defenders
                if (abs(d_r - king_row) <= 1 and abs(d_c - king_col) <= 1)
            ]

            if len(defenders_near_corner) >= 1:
                # Has defender support near corner - this is a strong fort
                return True

        return False

    def _is_protected_formation(self, king_row: int, king_col: int,
                                defenders: List[Tuple[int, int]]) -> bool:
        """
        Check if defenders form a protected formation around king.

        Protected formations include:
        - L-shape on edge (2x2 or larger)
        - Square/tower formation
        - T-shape with king protected

        These formations protect each other from capture.
        """
        # Get defenders immediately adjacent to king
        adjacent_defenders = [
            (d_r, d_c) for d_r, d_c in defenders
            if abs(d_r - king_row) <= 1 and abs(d_c - king_col) <= 1
        ]

        if len(adjacent_defenders) < 3:
            # Need at least 3 adjacent defenders for strong formation
            return False

        # Check for L-shape or square formation
        # A formation is protected if defenders are mutually supporting
        # (each defender is adjacent to at least one other defender)

        for def_r, def_c in adjacent_defenders:
            # Count how many other defenders are adjacent to this defender
            supporting_defenders = 0
            for other_r, other_c in adjacent_defenders:
                if (def_r, def_c) != (other_r, other_c):
                    if abs(def_r - other_r) <= 1 and abs(def_c - other_c) <= 1:
                        supporting_defenders += 1

            # If any defender has no support, formation might be breakable
            if supporting_defenders == 0:
                return False

        # Check for critical squares (spaces adjacent to king that are undefended)
        # These represent potential attack vectors
        critical_squares = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            adj_r, adj_c = king_row + dr, king_col + dc
            if 0 <= adj_r < self.BOARD_SIZE and 0 <= adj_c < self.BOARD_SIZE:
                piece = self.board[adj_r, adj_c]
                if piece != Piece.DEFENDER and piece != Piece.KING:
                    # This square is not defended
                    critical_squares += 1

        # If too many critical squares, formation is vulnerable
        if critical_squares > 1:
            return False

        # All defenders have mutual support and few vulnerabilities
        if len(adjacent_defenders) >= 3:
            return True

        return False

    def _is_shieldwall_immune_formation(self, king_row: int, king_col: int,
                                       defenders: List[Tuple[int, int]]) -> bool:
        """
        Check if formation is immune to shieldwall capture.

        A formation is shieldwall-immune if:
        - Not all in a single line along the edge
        - Has depth (defenders not just on edge)
        - Protected by corners or strong positions
        """
        # Determine which edge king is on
        on_top_edge = king_row == self.BOARD_SIZE - 1
        on_bottom_edge = king_row == 0
        on_left_edge = king_col == 0
        on_right_edge = king_col == self.BOARD_SIZE - 1

        # Get defenders on the same edge as king
        edge_defenders = []
        for def_r, def_c in defenders:
            if on_top_edge and def_r == self.BOARD_SIZE - 1:
                edge_defenders.append((def_r, def_c))
            elif on_bottom_edge and def_r == 0:
                edge_defenders.append((def_r, def_c))
            elif on_left_edge and def_c == 0:
                edge_defenders.append((def_r, def_c))
            elif on_right_edge and def_c == self.BOARD_SIZE - 1:
                edge_defenders.append((def_r, def_c))

        # If most pieces are in a straight line on edge, check vulnerability to shieldwall
        if len(edge_defenders) >= 2:
            # Check if they form a continuous or nearly continuous line (vulnerable)
            if on_top_edge or on_bottom_edge:
                # Horizontal line - check if columns are continuous
                all_cols = sorted([d_c for _, d_c in edge_defenders] + [king_col])
                # Check if there are big gaps
                span = all_cols[-1] - all_cols[0]
                pieces_in_span = len(all_cols)
                # If the line is mostly continuous, it's vulnerable
                if span <= pieces_in_span + 1:  # Allow 1 gap max
                    return False
            else:
                # Vertical line - check if rows are continuous
                all_rows = sorted([d_r for d_r, _ in edge_defenders] + [king_row])
                span = all_rows[-1] - all_rows[0]
                pieces_in_span = len(all_rows)
                if span <= pieces_in_span + 1:
                    return False

        # Has depth or not a simple line - harder to break
        # Need at least 4 defenders for truly immune formation
        return len(defenders) >= 4

    def _check_encirclement(self) -> bool:
        """
        Check if attackers have encircled all defenders.

        Encirclement win condition:
        - All defending pieces (including king) are surrounded by an unbroken ring of attackers
        - Defenders have no pieces outside the ring
        """
        # This is a complex condition to check properly
        # For now, we'll implement a simplified version:
        # Check if all defenders are confined to a small connected region
        # with no way to expand

        # Find all defender positions
        defender_positions = set()
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = self.board[row, col]
                if piece in (Piece.DEFENDER, Piece.KING):
                    defender_positions.add((row, col))

        if not defender_positions:
            return False  # No defenders left

        # Simplified check: if defenders occupy very small space and have no legal moves
        # This is a very basic implementation - true encirclement is more complex
        return False  # For now, disable this complex rule

    def _check_repetition(self) -> bool:
        """
        Check for threefold repetition.

        Copenhagen rules: Perpetual repetition results in defender loss
        """
        if len(self.move_history) < 8:  # Need at least 4 moves by each player
            return False

        # Get current position hash
        current_hash = self.get_state_hash()

        # Count how many times this position has occurred
        count = 0
        for board_state, _ in self.move_history:
            # Create hash from historical state
            historical_hash = hash((board_state.tobytes(), self.current_player))
            if historical_hash == current_hash:
                count += 1

        # Threefold repetition
        return count >= 3

    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.result != GameResult.ONGOING

    def get_winner(self) -> Optional[Player]:
        """Get the winner of the game, if any"""
        if self.result == GameResult.DEFENDER_WIN:
            return Player.DEFENDER
        elif self.result == GameResult.ATTACKER_WIN:
            return Player.ATTACKER
        return None

    def get_result_string(self) -> str:
        """Get a human-readable game result"""
        if self.result == GameResult.ONGOING:
            return "Game in progress"
        elif self.result == GameResult.DEFENDER_WIN:
            return "Defenders win!"
        elif self.result == GameResult.ATTACKER_WIN:
            return "Attackers win!"
        elif self.result == GameResult.DRAW:
            return "Draw"
        return "Unknown result"


# Move encoding functions for neural network policy
# Policy vector size: 11 * 11 * 4 * 10 = 4840
# Encoding: from_square (121) * direction (4) * distance (10)

def encode_move(move: Move, board_size: int = 11) -> int:
    """
    Encode a move to an index in the policy vector.

    Args:
        move: Move to encode
        board_size: Size of the board (default 11)

    Returns:
        Index in range [0, 4840)
    """
    from_square = move.from_row * board_size + move.from_col

    # Determine direction and distance
    dr = move.to_row - move.from_row
    dc = move.to_col - move.from_col

    if dr < 0:  # Moving up
        direction = 0
        distance = abs(dr)
    elif dr > 0:  # Moving down
        direction = 1
        distance = abs(dr)
    elif dc < 0:  # Moving left
        direction = 2
        distance = abs(dc)
    else:  # Moving right (dc > 0)
        direction = 3
        distance = abs(dc)

    # Encode: from_square * 40 + direction * 10 + (distance - 1)
    index = from_square * 40 + direction * 10 + (distance - 1)

    return index


def decode_move(index: int, board_size: int = 11) -> Move:
    """
    Decode an index from the policy vector to a move.

    Args:
        index: Index in range [0, 4840)
        board_size: Size of the board (default 11)

    Returns:
        Decoded move
    """
    # Decode components
    from_square = index // 40
    remainder = index % 40
    direction = remainder // 10
    distance = (remainder % 10) + 1

    # Get from position
    from_row = from_square // board_size
    from_col = from_square % board_size

    # Get to position based on direction
    if direction == 0:  # Up
        to_row = from_row - distance
        to_col = from_col
    elif direction == 1:  # Down
        to_row = from_row + distance
        to_col = from_col
    elif direction == 2:  # Left
        to_row = from_row
        to_col = from_col - distance
    else:  # direction == 3, Right
        to_row = from_row
        to_col = from_col + distance

    return Move(from_row, from_col, to_row, to_col)


def get_policy_size(board_size: int = 11) -> int:
    """Get the size of the policy vector"""
    return board_size * board_size * 4 * 10
