import itertools
import numpy as np


def clean_board(size):
    """
    Returns an empty tic tac toe board of given size.

    Args:
        size: The size of the side of the board.

    Returns:
        Numpy array of shape (size, size) containing only zeros.
    """
    return np.zeros((size, size), dtype=np.int)


def apply_move_inplace(board, position, side):
    """
    Applies move inplace without creating the copy.

    Args:
        board: The given board we want to apply the move to.
        position: The position we want to make the move in.
        side: The side we are making this move for, 1 for the first player, -1 for the second player.
    """
    if board[position] != 0:
        raise ValueError("The field of the board is already taken! Board{} = {}".format(position, board[position]))
    board[position] = side


def apply_move(board, position, side):
    """
    Returns a copy of the given board_state with the desired move applied.

    Args:
        board: The given board we want to apply the move to.
        position: The position we want to make the move in.
        side: The side we are making this move for, 1 for the first player, -1 for the second player.

    Returns:
        Numpy array of shape (size, size) representing the board after applied move.
    """
    board = np.array(board)
    apply_move_inplace(board, position, side)

    return board


def available_moves(board):
    """
    Returns all legal moves for the current board_state. These are all positions with value 0.

    Args:
        board: The given board we want to find all legal moves.

    Returns:
        List of tuples representing all possible positions to move.
    """
    return [tuple(position) for position in np.argwhere(board == 0).tolist()]


def diagonals_of_the_board_longer_equals_winning_length(board, winning_length):
    """
    Returns all diagonals of the board which are longer or equal given length.

    Args:
        board: The given board we want to find all diagonals.
        winning_length: shortest length of acceptable diagonal.

    Returns:
        list of lists: each sublist represents state of each acceptable length diagonal of the board.
    """
    diagonals = [board[::-1, :].diagonal(i) for i in range(-board.shape[0] + 1, board.shape[1])]
    diagonals.extend(board.diagonal(i) for i in range(board.shape[1] - 1, -board.shape[0], -1))
    return [diagonal.tolist() for diagonal in diagonals if diagonal.shape[0] >= winning_length]


def determine_line_winner(line, winning_length):
    """
    Determine if a player has won on the given line.

    Args:
        line: Line of the board we want to evaluate.
        winning_length: The number of moves in a row needed for a win.

    Returns:
        int: 1 if player one has won, -1 if player 2 has won, otherwise 0.
    """
    for k, g in itertools.groupby(line):
        if k != 0 and len(list(g)) >= winning_length:
            return k
    return 0


def determine_board_winner(board, winning_length):
    """
    Determine if a player has won on the given board.

    Args:
        board: The given board we want to determine winner.
        winning_length: The number of moves in a row needed for a win.

    Returns:
        int: 1 if player one has won, -1 if player 2 has won, otherwise 0.
    """

    for row in board:
        if determine_line_winner(row, winning_length):
            return row[0]

    for column in board.T:
        if determine_line_winner(column, winning_length):
            return column[0]

    for diagonal in diagonals_of_the_board_longer_equals_winning_length(board, winning_length):
        if determine_line_winner(diagonal, winning_length):
            return diagonal[0]

    return 0


def player_to_play(player1, player2, side_to_play):
    return player1 if side_to_play == 1 else player2


def play_game(size, winning_length, player1, player2):
    board = clean_board(size)
    side_to_play = 1
    legal_moves = available_moves(board)
    winner = 0

    while len(legal_moves) > 0 and not winner:
        player = player_to_play(player1, player2, side_to_play)
        board = apply_move(board, player(legal_moves), side_to_play)
        winner = determine_board_winner(board, winning_length)
        legal_moves = available_moves(board)
        side_to_play = -side_to_play

    return winner
