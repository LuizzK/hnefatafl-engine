"""
Unit tests for Copenhagen Hnefatafl game engine
"""

import pytest
import numpy as np
from hnefatafl.game import HnefataflGame, Piece, Player, Move, GameResult


class TestBoardSetup:
    """Test initial board setup"""

    def test_initial_board_has_correct_pieces(self):
        """Test that the initial board has the correct number of pieces"""
        game = HnefataflGame()

        # Count pieces
        attackers = np.count_nonzero(game.board == Piece.ATTACKER)
        defenders = np.count_nonzero(game.board == Piece.DEFENDER)
        kings = np.count_nonzero(game.board == Piece.KING)

        assert attackers == 24, "Should have 24 attackers"
        assert defenders == 12, "Should have 12 defenders"
        assert kings == 1, "Should have 1 king"

    def test_king_starts_at_throne(self):
        """Test that the king starts at the throne (center)"""
        game = HnefataflGame()
        assert game.board[5, 5] == Piece.KING

    def test_attackers_move_first(self):
        """Test that attackers have the first move"""
        game = HnefataflGame()
        assert game.current_player == Player.ATTACKER

    def test_corners_are_empty_initially(self):
        """Test that corner squares are empty at start"""
        game = HnefataflGame()
        for corner_r, corner_c in game.CORNERS:
            assert game.board[corner_r, corner_c] == Piece.EMPTY


class TestMovement:
    """Test piece movement"""

    def test_piece_can_move_horizontally(self):
        """Test that pieces can move horizontally"""
        game = HnefataflGame()
        # Find an attacker that can move
        # Attacker at (0, 3) can move horizontally
        moves = game._get_piece_moves(0, 3)
        assert len(moves) > 0

        # Check horizontal moves exist
        horizontal_moves = [m for m in moves if m.from_row == m.to_row]
        assert len(horizontal_moves) > 0

    def test_piece_can_move_vertically(self):
        """Test that pieces can move vertically"""
        game = HnefataflGame()
        # Attacker at (3, 0) can move vertically
        moves = game._get_piece_moves(3, 0)
        vertical_moves = [m for m in moves if m.from_col == m.to_col]
        assert len(vertical_moves) > 0

    def test_piece_cannot_jump_over_others(self):
        """Test that pieces cannot jump over other pieces"""
        game = HnefataflGame()
        # Place pieces to block movement
        # Clear a row and place pieces
        game.board[2, :] = Piece.EMPTY
        game.board[2, 3] = Piece.ATTACKER
        game.board[2, 5] = Piece.DEFENDER

        # Attacker at (2,3) should not be able to move past defender at (2,5)
        moves = game._get_piece_moves(2, 3)
        # Check that no move goes past column 4
        for move in moves:
            if move.from_row == 2 and move.to_row == 2:
                assert move.to_col < 5

    def test_only_king_can_land_on_corners(self):
        """Test that only the king can land on corner squares"""
        game = HnefataflGame()
        # Clear area around a corner
        game.board[0, 1] = Piece.EMPTY
        game.board[1, 0] = Piece.EMPTY
        game.board[1, 1] = Piece.ATTACKER

        # Attacker should not be able to move to corner
        moves = game._get_piece_moves(1, 1)
        corner_moves = [m for m in moves if (m.to_row, m.to_col) in game.CORNERS]
        assert len(corner_moves) == 0

    def test_pieces_can_pass_through_empty_throne(self):
        """Test that pieces can pass through the throne when it's empty"""
        game = HnefataflGame()
        # Clear a path through the throne
        game.board[:, :] = Piece.EMPTY
        game.board[5, 5] = Piece.KING  # Keep king somewhere

        # Place a piece on one side of throne with clear path on other side
        game.board[5, 3] = Piece.ATTACKER  # Attacker on left
        # Throne at (5,5) will be empty after we move king

        # Move king away from throne
        game.board[5, 5] = Piece.EMPTY
        game.board[6, 6] = Piece.KING  # Move king elsewhere

        moves = game._get_piece_moves(5, 3)
        # Should be able to move past throne (columns 6-10 are past throne at col 5)
        # Piece at col 3, throne at col 5, so moves to col > 5 means passed through throne
        through_throne = [m for m in moves if m.to_col > 5]
        assert len(through_throne) > 0


class TestCaptures:
    """Test piece capture mechanics"""

    def test_standard_capture_horizontal(self):
        """Test standard horizontal custodian capture"""
        game = HnefataflGame()
        # Set up a capture scenario
        game.board[:, :] = Piece.EMPTY
        game.board[5, 5] = Piece.KING  # Keep king on board

        # Attacker-Defender-Attacker horizontal
        game.board[3, 3] = Piece.ATTACKER
        game.board[3, 4] = Piece.DEFENDER
        game.board[3, 5] = Piece.ATTACKER

        game.current_player = Player.ATTACKER

        # Move one attacker to trigger capture (already in position, so move another piece then back)
        # Actually, let's simulate a move that creates the capture
        game.board[3, 5] = Piece.EMPTY
        game.board[2, 5] = Piece.ATTACKER

        # Make a move to create the capture
        move = Move(2, 5, 3, 5)
        game.make_move(move)

        # Defender at (3,4) should be captured
        assert game.board[3, 4] == Piece.EMPTY

    def test_standard_capture_vertical(self):
        """Test standard vertical custodian capture"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY
        game.board[5, 5] = Piece.KING

        # Attacker-Defender-Attacker vertical
        game.board[3, 3] = Piece.ATTACKER
        game.board[4, 3] = Piece.DEFENDER
        game.board[5, 3] = Piece.EMPTY
        game.board[6, 3] = Piece.ATTACKER

        game.current_player = Player.ATTACKER

        # Move to create capture
        move = Move(5, 3, 5, 3)  # This won't work, let me fix
        game.board[4, 4] = Piece.ATTACKER
        move = Move(4, 4, 5, 3)
        game.board[5, 3] = Piece.ATTACKER  # Place directly

        game._check_standard_captures(5, 3)

        # Defender at (4,3) should be captured
        assert game.board[4, 3] == Piece.EMPTY

    def test_capture_against_throne(self):
        """Test capture using throne as hostile square"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY
        game.board[5, 5] = Piece.KING

        # Defender next to throne, attacker on other side
        game.board[5, 4] = Piece.DEFENDER
        game.board[5, 6] = Piece.KING  # Move king
        game.board[5, 5] = Piece.EMPTY  # Throne empty

        game.board[6, 4] = Piece.ATTACKER
        game.current_player = Player.ATTACKER

        # Move attacker to create capture against throne
        game.board[4, 4] = Piece.ATTACKER
        game._check_standard_captures(4, 4)

        # Defender should be captured against throne
        assert game.board[5, 4] == Piece.EMPTY

    def test_king_requires_four_side_capture(self):
        """Test that king requires 4-sided capture"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # Place king in middle of board
        game.board[5, 5] = Piece.KING

        # Surround with 3 attackers (not enough)
        game.board[4, 5] = Piece.ATTACKER
        game.board[5, 4] = Piece.ATTACKER
        game.board[6, 5] = Piece.ATTACKER

        game._check_king_capture()
        assert game.board[5, 5] == Piece.KING  # King still there

        # Add 4th attacker
        game.board[5, 6] = Piece.ATTACKER
        game._check_king_capture()
        assert game.board[5, 5] == Piece.EMPTY  # King captured

    def test_king_cannot_be_captured_on_edge(self):
        """Test that king cannot be captured on board edge"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # Place king on edge
        game.board[0, 5] = Piece.KING

        # Surround with attackers (except bottom edge)
        game.board[0, 4] = Piece.ATTACKER
        game.board[0, 6] = Piece.ATTACKER
        game.board[1, 5] = Piece.ATTACKER

        game._check_king_capture()
        assert game.board[0, 5] == Piece.KING  # King safe on edge


class TestWinConditions:
    """Test win conditions"""

    def test_king_reaching_corner_wins(self):
        """Test that king reaching corner is a defender win"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # Place king near corner
        game.board[0, 1] = Piece.KING
        game.current_player = Player.DEFENDER

        # Move king to corner
        move = Move(0, 1, 0, 0)
        game.make_move(move)

        assert game.result == GameResult.DEFENDER_WIN

    def test_king_capture_wins_for_attackers(self):
        """Test that capturing king wins for attackers"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # Place king and surrounding attackers
        game.board[5, 5] = Piece.KING
        game.board[4, 5] = Piece.ATTACKER
        game.board[5, 4] = Piece.ATTACKER
        game.board[6, 5] = Piece.ATTACKER
        game.board[5, 7] = Piece.ATTACKER

        game.current_player = Player.ATTACKER

        # Move attacker to capture king
        move = Move(5, 7, 5, 6)
        game.make_move(move)

        assert game.result == GameResult.ATTACKER_WIN

    def test_no_legal_moves_loses(self):
        """Test that having no legal moves results in a loss"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # Place king in corner surrounded by attackers
        game.board[1, 1] = Piece.KING
        game.board[0, 1] = Piece.ATTACKER
        game.board[1, 0] = Piece.ATTACKER
        game.board[1, 2] = Piece.ATTACKER
        game.board[2, 1] = Piece.ATTACKER

        game.current_player = Player.DEFENDER
        game._check_game_end()

        # Defender has no moves, should lose
        assert game.result == GameResult.ATTACKER_WIN


class TestStateEncoding:
    """Test state encoding for neural network"""

    def test_encode_state_shape(self):
        """Test that encoded state has correct shape"""
        game = HnefataflGame()
        state = game.encode_state()
        assert state.shape == (15, 11, 11)

    def test_encode_state_has_pieces(self):
        """Test that encoded state includes piece positions"""
        game = HnefataflGame()
        state = game.encode_state()

        # Current player is attacker, so channel 0 should have attackers
        assert np.sum(state[0]) > 0  # Has attacker pieces

        # Channel 2 should have defenders (opponent)
        assert np.sum(state[2]) > 0  # Has defender pieces

    def test_encode_state_has_special_squares(self):
        """Test that encoded state includes special squares"""
        game = HnefataflGame()
        state = game.encode_state()

        # Channel 4: Throne
        assert state[4, 5, 5] == 1.0

        # Channel 5: Corners
        for corner_r, corner_c in game.CORNERS:
            assert state[5, corner_r, corner_c] == 1.0


def test_game_copy():
    """Test that game state can be copied"""
    game = HnefataflGame()
    game_copy = game.copy()

    # Modify copy
    game_copy.current_player = Player.DEFENDER

    # Original should be unchanged
    assert game.current_player == Player.ATTACKER
    assert game_copy.current_player == Player.DEFENDER


def test_game_string_representation():
    """Test that game can be printed"""
    game = HnefataflGame()
    game_str = str(game)

    assert 'Attacker' in game_str or 'Defender' in game_str
    assert 'a b c d e f g h i j k' in game_str
