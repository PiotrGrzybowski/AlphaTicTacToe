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


def _evaluate_line(line, winning_length):
    count = 0
    last_side = 0
    score = 0
    neutrals = 0

    for x in line:
        if x == last_side:
            count += 1
            if count == winning_length and neutrals == 0:
                return 100000 * x  # a side has already won here
        elif x == 0:  # we could score here
            neutrals += 1
        elif x == -last_side:
            if neutrals + count >= winning_length:
                score += (count - 1) * last_side
            count = 1
            last_side = x
            neutrals = 0
        else:
            last_side = x
            count = 1

    if neutrals + count >= winning_length:
        score += (count - 1) * last_side

    return score


def evaluate(board_state, winning_length):
    """An evaluation function for this game, gives an estimate of how good the board position is for the plus player.
    There is no specific range for the values returned, they just need to be relative to each other.

    Args:
        winning_length (int): The length needed to win a game
        board_state (tuple): State of the board

    Returns:
        number
    """
    board_width = len(board_state)
    board_height = len(board_state[0])

    score = 0

    # check rows
    for x in range(board_width):
        score += _evaluate_line(board_state[x], winning_length)
    # check columns
    for y in range(board_height):
        score += _evaluate_line((i[y] for i in board_state), winning_length)

    # check diagonals
    diagonals_start = -(board_width - winning_length)
    diagonals_end = (board_width - winning_length)
    for d in range(diagonals_start, diagonals_end + 1):
        score += _evaluate_line(
            (board_state[i][i + d] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)
    for d in range(diagonals_start, diagonals_end + 1):
        score += _evaluate_line(
            (board_state[i][board_height - i - d - 1] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)

    return score


def hash_board(board):
    return ''.join(map(str, (itertools.chain(*board.tolist()))))


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


def generate_boards(board, side_to_play):
    legal_moves = available_moves(board)
    winner = determine_board_winner(board, 3)
    if len(legal_moves) == 0 or winner != 0:
        return [board.tolist()]
    else:
        result = []
        for move in legal_moves:
            new_board = apply_move(board, move, side_to_play)
            result_board = generate_boards(new_board, -side_to_play)
            if new_board.tolist() not in result_board:
                result.append(new_board.tolist())

            result += result_board
        return result


def generate_winners(board, side_to_play):
    legal_moves = available_moves(board)
    winner = determine_board_winner(board, 3)
    if len(legal_moves) == 0 or winner != 0:
        return [winner]
    else:
        result = []
        for move in legal_moves:
            new_board = apply_move(board, move, side_to_play)
            result += generate_winners(new_board, -side_to_play)
    return result


if __name__ == '__main__':
    # board = clean_board(3)
    board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    side_to_play = -1
    # print("{} -> {}".format(board.tolist(), np.mean(generate_winners(np.array(board), side_to_play))))
    states = generate_boards(board, side_to_play)
    import itertools
    states_set = sorted(states, reverse=True)
    states_set = list(k for k, _ in itertools.groupby(states_set))

    print(len(states_set))

    # for s in states_set:
    #     print("{} -> {}".format(s, np.mean(generate_winners(np.array(s), side_to_play))))

    X = np.array(states_set).reshape(len(states_set), 9)
    Y = np.array([np.mean(generate_winners(np.array(s), side_to_play)) for s in states_set]).reshape(len(states_set), 1)

    print(X.shape)
    print(Y.shape)

    np.save('X.npy', X)
    np.save('Y.npy', Y)
