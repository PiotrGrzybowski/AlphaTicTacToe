import collections
import random

import numpy as np
import tensorflow as tf

# from game.tic_tac_toe import flat_move_to_tuple
from game.tic_tac_toe import flat_move_to_tuple, playya_game
from players.random_player import RandomPlayer
from utils.network_utils import create_network, get_deterministic_network_move


def train_policy_gradients(layers, learning_rate, games, log_every, winning_length, opponent, batch_size):
    reward_tf = tf.placeholder(tf.float32, shape=(None,))
    actual_move = tf.placeholder(tf.float32, shape=(None, 9))

    input_layer, output_layer, variables = create_network(layers)
    policy_gradient = -tf.log(tf.reduce_sum(tf.multiply(actual_move, output_layer), reduction_indices=1)) * reward_tf
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(policy_gradient)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        boards_batch, moves_batch, rewards_batch = [], [], []
        results = collections.deque(maxlen=log_every)

        def training_move(board, side):
            boards_batch.append(np.ravel(board) * side)
            move = get_deterministic_network_move(session, input_layer, output_layer, board, side)
            moves_batch.append(move)

            return flat_move_to_tuple(board, move.argmax())

        for episode_number in range(1, games):
            # randomize if going first or second
            if bool(random.getrandbits(1)):
                # print("Starts MLP with 1")
                reward = playya_game(3, training_move, opponent.get_move)
            else:
                # print("Starts Random with 1")
                reward = -playya_game(3, opponent.get_move, training_move)

            results.append(reward)
            last_game_length = len(boards_batch) - len(rewards_batch)
            reward /= float(last_game_length)

            rewards_batch += ([reward] * last_game_length)

            if episode_number % batch_size == 0:
                normalized_rewards = normalize_rewards(rewards_batch)

                np_mini_batch_board_states = np.array(boards_batch).reshape(len(rewards_batch),
                                                                            *input_layer.get_shape().as_list()[1:])

                session.run(optimizer, feed_dict={input_layer: np_mini_batch_board_states,
                                                  reward_tf: normalized_rewards,
                                                  actual_move: moves_batch})

                boards_batch, moves_batch, rewards_batch = [], [], []
            if episode_number % log_every == 0:
                print("episode: %s win_rate: %s" % (episode_number, _win_rate(log_every, results)))


def _win_rate(print_results_every, results):
    i = sum(results)
    every___ = (print_results_every * 2.)
    return 0.5 + i / every___


def normalize_rewards(rewards_batch):
    normalized_rewards = rewards_batch - np.mean(rewards_batch)
    rewards_std = np.std(normalized_rewards)
    if rewards_std != 0:
        normalized_rewards /= rewards_std
    else:
        print("warning: got mini batch std of 0.")
    return normalized_rewards


if __name__ == '__main__':
    train_policy_gradients(layers=[9, 100, 100, 100, 9],
                           learning_rate=1e-4,
                           batch_size=100,
                           games=100000,
                           log_every=1000,
                           opponent=RandomPlayer(-1),
                           winning_length=3)
