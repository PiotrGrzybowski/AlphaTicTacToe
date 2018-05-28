from game.tic_tac_toe import play_game
from players.random_player import random_play

if __name__ == '__main__':
    results = [play_game(3, 3, random_play, random_play) for _ in range(10000)]
    print("Win: {}".format(results.count(1)))
    print("Draw: {}".format(results.count(0)))
    print("Loss: {}".format(results.count(-1)))

