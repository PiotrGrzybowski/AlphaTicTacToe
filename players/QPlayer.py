import numpy as np

from game.tic_tac_toe import hash_board, available_moves, determine_board_winner, apply_move, clean_board, evaluate
from players.random_player import RandomPlayer


def dict_max_key(dictionary):
    return max(dictionary.keys(), key=lambda key: dictionary[key])


class QPlayer:
    def __init__(self, side, winning_length):
        self.q_table = {}
        self.side = side
        self.winning_length = winning_length

    def add_board(self, board):
        board_hash = hash_board(board)
        if self.q_table.get(board_hash) is None:
            legal_moves = available_moves(board)
            self.q_table[board_hash] = {move: 1.0 for move in legal_moves}

        return board_hash

    def get_move(self, board):
        board_hash = self.add_board(board)
        return dict_max_key(self.q_table[board_hash])

    def calculate_reward(self, board):
        return determine_board_winner(board, self.winning_length)

    def learn_q(self, board, move):
        board_hash = self.add_board(board)
        new_board = apply_move(board, move, self.side)
        new_board_hash = self.add_board(new_board)

        reward = self.calculate_reward(new_board)

        if reward != 0 or len(available_moves(new_board_hash)) == 0:
            expected = reward
        else:
            expected_rewards = self.q_table[new_board_hash]

            expected = reward + (0.9 * max(expected_rewards.values()))

        change = 0.3 * (expected - self.q_table[board_hash][move])
        self.q_table[board_hash][move] += change


def swap_players(p1, p2):
    return p2, p1


if __name__ == '__main__':
    qplayer = QPlayer(1, 3)
    rplayer = RandomPlayer(-1)
    winning_length = 3
    games = 10000

    results = {1: 0,
               0: 0,
               -1: 0}
    import tqdm
    import pickle

    # for i in tqdm.tqdm(range(games)):
    #     board = clean_board(3)
    #     winner = 0
    #     # print(i)
    #
    #     while True:
    #         move = qplayer.get_move(board)
    #         qplayer.learn_q(board, move)
    #         board = apply_move(board, move, qplayer.side)
    #         winner = determine_board_winner(board, winning_length)
    #
    #         if winner != 0 or len(available_moves(board)) == 0:
    #             break
    #
    #         move = rplayer.get_move(board)
    #         board = apply_move(board, move, rplayer.side)
    #         winner = determine_board_winner(board, winning_length)
    #
    #         if winner != 0 or len(available_moves(board)) == 0:
    #             break
    #
    #     results[winner] += 1
    #
    # print()
    # for k, v in results.items():
    #     print("{}: {}".format(k, v))
    #
    # with open('qlr.pkl', 'wb') as handle:
    #     pickle.dump(qplayer, handle)

    with open('qlr.pkl', 'rb') as handle:
        qplayer = pickle.load(handle)
        board = clean_board(3)
        print(board)
        print(evaluate(board, 3))
        print()
        while True:
            move = qplayer.get_move(board)
            board = apply_move(board, move, qplayer.side)
            print(board)
            print(evaluate(board, 3))
            print()
            m = int(input())
            move = (m // 10, m % 10)
            board = apply_move(board, move, -1)
            print(board)
            print(evaluate(board, 3))
            print()

