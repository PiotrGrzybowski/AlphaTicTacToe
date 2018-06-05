import numpy as np

from keras.models import model_from_json

from game.tic_tac_toe import available_moves, apply_move, clean_board


class MlpPlayer:
    def __init__(self, side_to_play):
        self.model = self.load_model()
        self.side_to_play = side_to_play

    def min_max_best_move(self, evaluations):
        if self.side_to_play == 1:
            return evaluations.argmax()
        else:
            return evaluations.argmin()

    def get_move(self, board):
        legal_moves = available_moves(board)
        new_boards = np.array([apply_move(board, move, self.side_to_play) for move in legal_moves])
        # print(new_boards.reshape(len(legal_moves), 9))
        # possible_boards = [apply_move(board, move, self.side_to_play) for move in legal_moves]
        evaluations = self.model.predict(new_boards.reshape(len(legal_moves), 9))
        print(legal_moves)
        print(evaluations)
        return legal_moves[self.min_max_best_move(evaluations)]

    def load_model(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        return loaded_model


if __name__ == '__main__':
    mlp_player = MlpPlayer(side_to_play=1)
    board = np.array([[1, -1, 1], [-1, 1, -1], [0, 0, 0]])
    board = np.array([[-1, -1, 0], [0, 1, 0], [0, 0, 1]])
    print(board)
    mlp_player.get_move(board)
    # mlp_player.model.predict(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    board = clean_board(3)
    print(board)
    print()
    while True:
        move = mlp_player.get_move(board)
        board = apply_move(board, move, mlp_player.side_to_play)
        print(board)
        # print(evaluate(board, 3))
        print()
        m = int(input())
        move = (m // 10, m % 10)
        board = apply_move(board, move, -1)
        print(board)
        # print(evaluate(board, 3))
        print()
