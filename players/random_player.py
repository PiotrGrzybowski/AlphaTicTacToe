import random

from game.tic_tac_toe import available_moves


def random_play(legal_moves):
    return random.choice(legal_moves)


class RandomPlayer:
    def __init__(self, side):
        self.side = side

    def get_move(self, board):
        return random.choice(available_moves(board))