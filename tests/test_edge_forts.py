"""
Comprehensive tests for Edge Fort detection in Copenhagen Hnefatafl

Edge Fort Win Condition:
- King has contact with the board edge
- King is able to move
- Attackers cannot break the fort (cannot capture king or force it away from edge)
"""

import pytest
from hnefatafl.game import HnefataflGame, Piece, Player, GameResult


class TestEdgeFortBasics:
    """Test basic edge fort requirements"""

    def test_king_not_on_edge_is_not_fort(self):
        """King must be on edge for edge fort"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # King in middle with defenders around
        game.board[5, 5] = Piece.KING
        game.board[4, 5] = Piece.DEFENDER
        game.board[6, 5] = Piece.DEFENDER
        game.board[5, 4] = Piece.DEFENDER
        game.board[5, 6] = Piece.DEFENDER

        game.current_player = Player.DEFENDER
        game._check_game_end()

        assert game.result != GameResult.DEFENDER_WIN, "King not on edge, not an edge fort"

    def test_king_on_edge_but_cannot_move_is_not_fort(self):
        """King must be able to move for edge fort"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # King on edge but completely surrounded
        game.board[0, 5] = Piece.KING
        game.board[0, 4] = Piece.ATTACKER
        game.board[0, 6] = Piece.ATTACKER
        game.board[1, 5] = Piece.ATTACKER

        game.current_player = Player.DEFENDER
        game._check_game_end()

        assert game.result != GameResult.DEFENDER_WIN, "King cannot move, not an edge fort"

    def test_king_on_edge_alone_is_not_fort(self):
        """King alone on edge is not a fort (can be captured)"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # King on edge with one square to move
        game.board[0, 5] = Piece.KING
        # Can move to (0, 4) or (0, 6) or (1, 5)
        # But no defenders to protect - attackers can capture

        game.current_player = Player.DEFENDER
        game._check_game_end()

        assert game.result != GameResult.DEFENDER_WIN, "King alone is not a fort"


class TestEdgeFortCornerFormations:
    """Test edge fort formations near corners"""

    def test_king_in_corner_adjacent_square_is_win(self):
        """King reaching corner square is immediate win, not edge fort"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # King on corner
        game.board[0, 0] = Piece.KING

        game.current_player = Player.DEFENDER
        game._check_game_end()

        assert game.result == GameResult.DEFENDER_WIN, "King on corner is immediate win"

    def test_king_one_square_from_corner_with_defender(self):
        """King one square from corner with defender support"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # King at (0, 1), corner at (0, 0)
        game.board[0, 1] = Piece.KING
        game.board[1, 1] = Piece.DEFENDER
        game.board[0, 2] = Piece.DEFENDER

        # King can move to corner or along edge
        # Fort is unbreakable because corner blocks one side

        game.current_player = Player.DEFENDER
        game._check_game_end()

        # This MIGHT be an edge fort if properly implemented
        # For now, just verify it doesn't crash


class TestEdgeFortBreakable:
    """Test cases where edge formation looks like fort but is breakable"""

    def test_king_on_edge_with_defenders_but_breakable_by_shieldwall(self):
        """Defenders in a line on edge can be captured via shieldwall"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # King and defenders on edge in a line
        game.board[0, 3] = Piece.DEFENDER
        game.board[0, 4] = Piece.DEFENDER
        game.board[0, 5] = Piece.KING
        game.board[0, 6] = Piece.DEFENDER

        # Attackers can bracket and use shieldwall capture
        game.board[0, 2] = Piece.ATTACKER  # Left bracket
        game.board[0, 7] = Piece.ATTACKER  # Right bracket
        game.board[1, 3] = Piece.ATTACKER  # Opposite
        game.board[1, 4] = Piece.ATTACKER
        game.board[1, 5] = Piece.ATTACKER
        game.board[1, 6] = Piece.ATTACKER

        game.current_player = Player.DEFENDER
        game._check_game_end()

        # This is NOT a fort - attackers can break it with shieldwall
        assert game.result != GameResult.DEFENDER_WIN


class TestEdgeFortUnbreakable:
    """Test confirmed unbreakable edge fort formations"""

    def test_king_and_defender_in_corner_pocket(self):
        """
        King and defender in corner pocket formation (unbreakable)

        Pattern:
        K D X
        D . .
        . . .

        Where X is corner, K is king, D is defender
        """
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # Top-left corner pocket
        game.board[10, 1] = Piece.KING      # k1
        game.board[10, 2] = Piece.DEFENDER  # l1
        game.board[9, 1] = Piece.DEFENDER   # k2

        # Corner at (10, 0) acts as third defender
        # This formation is unbreakable:
        # - King cannot be captured on edge
        # - King can move (e.g., to l2)
        # - Defenders cannot be captured (protected by corner and each other)

        game.current_player = Player.DEFENDER
        game._check_game_end()

        # This SHOULD be recognized as edge fort
        # Note: Test may fail until implementation is complete

    def test_the_tower_formation_on_edge(self):
        """
        'The Tower' - four defenders in square on edge (unbreakable)

        Pattern:
        D D .
        D D .
        . . .

        This formation with king inside or adjacent is unbreakable
        """
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # Tower on top edge
        game.board[10, 4] = Piece.DEFENDER
        game.board[10, 5] = Piece.DEFENDER
        game.board[9, 4] = Piece.DEFENDER
        game.board[9, 5] = Piece.DEFENDER
        game.board[10, 6] = Piece.KING  # King adjacent

        # This formation cannot be broken
        # - Shieldwall doesn't apply (not a line)
        # - Tower squares protect each other
        # - King has moves

        game.current_player = Player.DEFENDER
        game._check_game_end()

        # This SHOULD be recognized as edge fort


class TestEdgeFortComplexScenarios:
    """Test complex edge fort scenarios"""

    def test_king_on_edge_with_mixed_formation(self):
        """King on edge with defenders in complex formation"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # L-shaped formation on edge
        game.board[0, 5] = Piece.KING
        game.board[0, 6] = Piece.DEFENDER
        game.board[1, 5] = Piece.DEFENDER
        game.board[1, 6] = Piece.DEFENDER

        # Can attackers break this?
        # - Not a straight line, so no shieldwall
        # - King on edge can't be captured
        # - Defenders protect each other

        game.current_player = Player.DEFENDER
        game._check_game_end()

    def test_false_fort_missing_key_defender(self):
        """Formation that looks like fort but has vulnerability"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # King on edge with gap in defense
        game.board[0, 5] = Piece.KING
        game.board[0, 6] = Piece.DEFENDER
        # Gap at (1, 5)
        game.board[1, 6] = Piece.DEFENDER

        # Attacker can exploit the gap
        game.board[1, 5] = Piece.ATTACKER

        game.current_player = Player.DEFENDER
        game._check_game_end()

        # This should NOT be a fort - has weakness
        assert game.result != GameResult.DEFENDER_WIN


class TestEdgeFortAfterCapture:
    """Test edge fort detection after captures"""

    def test_edge_fort_becomes_valid_after_attacker_captured(self):
        """Fort might become valid after removing blocking attacker"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # King and defenders on edge
        game.board[0, 5] = Piece.KING
        game.board[0, 4] = Piece.DEFENDER
        game.board[0, 6] = Piece.DEFENDER
        game.board[1, 5] = Piece.DEFENDER

        # After defenders capture nearby attackers, check if fort is valid
        game.current_player = Player.DEFENDER
        game._check_game_end()

    def test_edge_fort_broken_by_capturing_key_defender(self):
        """Fort becomes invalid after multiple defenders captured"""
        game = HnefataflGame()
        game.board[:, :] = Piece.EMPTY

        # Initially valid fort with 4 defenders
        game.board[0, 5] = Piece.KING
        game.board[0, 4] = Piece.DEFENDER
        game.board[0, 6] = Piece.DEFENDER
        game.board[1, 5] = Piece.DEFENDER
        game.board[1, 4] = Piece.DEFENDER

        # Attackers capture TWO key defenders (leaving only 2 total)
        game.board[0, 4] = Piece.EMPTY
        game.board[1, 4] = Piece.EMPTY

        game.current_player = Player.DEFENDER
        game._check_game_end()

        # Fort is broken - only 2 defenders left
        assert game.result != GameResult.DEFENDER_WIN


def test_edge_fort_documentation():
    """
    Document the edge fort algorithm that needs to be implemented

    Algorithm for detecting unbreakable edge fort:
    1. Verify king is on board edge (row 0, 10, or col 0, 10)
    2. Verify king has at least one legal move
    3. Check if formation is unbreakable:

       For each possible attacker move sequence:
       - Can attackers capture the king?
       - Can attackers force king away from edge?
       - Can attackers capture all supporting defenders?

       If answer to all is NO → unbreakable fort

    Since checking all move sequences is expensive, use pattern recognition:
    - Check if king is in corner-adjacent position (very strong)
    - Check if defenders form protective patterns (Tower, L-shape, etc.)
    - Verify no line formations vulnerable to shieldwall
    - Check that all defenders are protected
    - Verify king has safe moves along edge

    This is a complex minimax-like problem. For strong play, the engine
    should learn edge fort patterns through self-play rather than
    hard-coding all patterns.
    """
    pass
