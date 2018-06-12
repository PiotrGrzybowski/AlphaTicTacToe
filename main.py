from game.tic_tac_toe import play_game, playya_game
from players.random_player import random_play, RandomPlayer
import numpy as np

if __name__ == '__main__':
    results = [play_game(3, 3, random_play, random_play) for _ in range(10000)]
    print("Win: {}".format(results.count(1)))
    print("Draw: {}".format(results.count(0)))
    print("Loss: {}".format(results.count(-1)))

    # board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # side_to_play = 1
    # playya_game(board, RandomPlayer(1), RandomPlayer(-1), True, 3)

