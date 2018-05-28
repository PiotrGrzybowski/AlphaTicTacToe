import numpy as np
import unittest

from game.tic_tac_toe import apply_move, apply_move_inplace, clean_board, available_moves, determine_line_winner, \
    diagonals_of_the_board_longer_equals_winning_length, determine_board_winner


class TestTicTacToe(unittest.TestCase):
    def test_new_board_shape(self):
        size = 3
        board = clean_board(size)

        self.assertEqual(board.shape, (size, size))

    def test_new_board_is_empty(self):
        size = 3
        board = clean_board(size)

        self.assertEqual(np.sum(board), 0)

    def test_apply_move_inplace_on_empty_field(self):
        size = 3
        board = clean_board(size)
        position = (0, 0)

        apply_move_inplace(board, position, 1)

        self.assertEqual(board[position], 1)

    def test_apply_move_inplace_on_taken_field(self):
        size = 3
        board = clean_board(size)
        position = (0, 0)

        apply_move_inplace(board, position, 1)
        self.assertRaises(ValueError, apply_move_inplace, board, position, -1)

    def test_apply_move_on_empty_field(self):
        size = 3
        board = clean_board(size)
        position = (0, 0)

        new_board = apply_move(board, position, 1)

        self.assertEqual(board[position], 0)
        self.assertEqual(new_board[position], 1)

    def test_apply_move_on_taken_field(self):
        size = 3
        board = clean_board(size)
        position = (0, 0)

        apply_move_inplace(board, position, 1)
        self.assertRaises(ValueError, apply_move, board, position, -1)

    def test_available_moves(self):
        size = 3
        board = clean_board(size)

        self.assertEqual(len(available_moves(board)), 9)

        apply_move_inplace(board, (0, 0), 1)
        self.assertEqual(len(available_moves(board)), 8)

        board = np.ones((size, size), dtype=np.int)
        self.assertEqual(len(available_moves(board)), 0)

    def test_is_line_winning_on_empty_line(self):
        size = 3
        winning_length = 3
        board = clean_board(size)

        self.assertFalse(determine_line_winner(board[0, :], winning_length))
        self.assertEqual(determine_line_winner(board[0, :], winning_length), 0)

    def test_is_line_winning_on_winning_line(self):
        size = 3
        winning_length = 3
        board = clean_board(size)
        side = -1

        apply_move_inplace(board, (0, 0), side)
        apply_move_inplace(board, (0, 1), side)
        apply_move_inplace(board, (0, 2), side)

        self.assertTrue(determine_line_winner(board[0, :], winning_length))
        self.assertEqual(determine_line_winner(board[0, :], winning_length), side)

    def test_diagonals_of_the_board(self):
        board = np.arange(0, 9).reshape((3, 3))
        length = 3

        self.assertEqual(len(diagonals_of_the_board_longer_equals_winning_length(board, length)), 2)

    def test_is_diagonal_winning_on_empty_diagonals(self):
        size = 3
        winning_length = 3
        board = clean_board(size)
        side = -1
        diagonals = diagonals_of_the_board_longer_equals_winning_length(board, 3)

        self.assertFalse(determine_line_winner(diagonals[0], winning_length))
        self.assertEqual(determine_line_winner(diagonals[0], winning_length), 0)

        self.assertFalse(determine_line_winner(diagonals[1], winning_length))
        self.assertEqual(determine_line_winner(diagonals[1], winning_length), 0)

        apply_move_inplace(board, (0, 0), side)
        apply_move_inplace(board, (1, 1), side)
        apply_move_inplace(board, (2, 2), side)
        diagonals = diagonals_of_the_board_longer_equals_winning_length(board, 3)
        self.assertTrue(determine_line_winner(diagonals[1], winning_length))
        self.assertEqual(determine_line_winner(diagonals[1], winning_length), side)

    def test_is_diagonal_winning_on_winning_diagonal(self):
        size = 3
        winning_length = 3
        board = clean_board(size)
        side = -1

        apply_move_inplace(board, (0, 0), side)
        apply_move_inplace(board, (1, 1), side)
        apply_move_inplace(board, (2, 2), side)
        diagonals = diagonals_of_the_board_longer_equals_winning_length(board, 3)

        self.assertTrue(determine_line_winner(diagonals[1], winning_length))
        self.assertEqual(determine_line_winner(diagonals[1], winning_length), side)

    def test_determine_board_winner_on_empty_board(self):
        size = 3
        winning_length = 3
        board = clean_board(size)

        self.assertEqual(determine_board_winner(board, winning_length), 0)

    def test_determine_board_winner_on_winning_row(self):
        size = 3
        winning_length = 3
        board = clean_board(size)
        side = -1

        apply_move_inplace(board, (0, 0), side)
        apply_move_inplace(board, (0, 1), side)
        apply_move_inplace(board, (0, 2), side)

        self.assertEqual(determine_board_winner(board, winning_length), side)

    def test_determine_board_winner_on_winning_column(self):
        size = 3
        winning_length = 3
        board = clean_board(size)
        side = -1

        apply_move_inplace(board, (0, 0), side)
        apply_move_inplace(board, (1, 0), side)
        apply_move_inplace(board, (2, 0), side)

        self.assertEqual(determine_board_winner(board, winning_length), side)

    def test_determine_board_winner_on_winning_diagonal(self):
        size = 3
        winning_length = 3
        board = clean_board(size)
        side = -1

        apply_move_inplace(board, (0, 0), side)
        apply_move_inplace(board, (1, 1), side)
        apply_move_inplace(board, (2, 2), side)

        self.assertEqual(determine_board_winner(board, winning_length), side)

    def test_determine_board_winner_on_draw_board(self):
        board = np.array([[1, -1, 1], [-1, 1, -1], [-1, 1, -1]])
        winning_length = 3

        self.assertEqual(determine_board_winner(board, winning_length), 0)


if __name__ == '__main__':
    unittest.main()
