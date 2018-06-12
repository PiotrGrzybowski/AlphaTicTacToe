import tensorflow as tf
import numpy as np


def create_network(layers):
    inputs = layers[0]
    hidden = layers[:1: -1]
    outputs = layers[-1]
    variables = []
    last_layer = None
    last_layer_units = None

    with tf.name_scope('network'):
        input_layer = tf.placeholder(dtype=tf.float32, shape=(None, inputs))
        last_layer = input_layer

        for hidden_units in hidden:
            last_layer_units = int(last_layer.get_shape()[-1])
            hidden_weights = tf.Variable(initialize_weights(hidden_units, last_layer_units), name='weights')
            hidden_bias = tf.Variable(initialize_bias(hidden_units), name='biases')

            variables.append(hidden_weights)
            variables.append(hidden_bias)

            last_layer = tf.nn.relu(tf.matmul(last_layer, hidden_weights) + hidden_bias)

        output_weights = tf.Variable(initialize_weights(outputs, last_layer_units), name="output_weights")
        output_bias = tf.Variable(initialize_bias(outputs), name="output_bias")

        variables.append(output_weights)
        variables.append(output_bias)

        output_layer = tf.nn.softmax(tf.matmul(last_layer, output_weights) + output_bias)

    return input_layer, output_layer, variables


def initialize_bias(units):
    return tf.constant(0.01, shape=(units,))


def initialize_weights(units, last_layer_units):
    return tf.truncated_normal((last_layer_units, units), stddev=1. / np.sqrt(last_layer_units))


def get_deterministic_network_move(session, input_layer, output_layer, board_state, side, valid_only=False,
                                   game_spec=None):
    """Choose a move for the given board_state using a deterministic policy. A move is selected using the values from
    the output_layer and selecting the move with the highest score.

    Args:
        session (tf.Session): Session used to run this network
        input_layer (tf.Placeholder): Placeholder to the network used to feed in the board_state
        output_layer (tf.Tensor): Tensor that will output the probabilities of the moves, we expect this to be of
            dimesensions (None, board_squares).
        board_state: The board_state we want to get the move for.
        side: The side that is making the move.

    Returns:
        (np.array) It's shape is (board_squares), and it is a 1 hot encoding for the move the network has chosen.
    """
    np_board_state = np.array(board_state)
    np_board_state = np_board_state.reshape(1, *input_layer.get_shape().as_list()[1:])
    if side == -1:
        np_board_state = -np_board_state

    probability_of_actions = session.run(output_layer,
                                         feed_dict={input_layer: np_board_state})[0]

    if valid_only:
        available_moves = game_spec.available_moves(board_state)
        available_moves_flat = [game_spec.tuple_move_to_flat(x) for x in available_moves]
        for i in range(game_spec.board_squares()):
            if i not in available_moves_flat:
                probability_of_actions[i] = 0

    move = np.argmax(probability_of_actions)
    one_hot = np.zeros(len(probability_of_actions))
    one_hot[move] = 1.
    return one_hot


if __name__ == '__main__':
    print(3)