import threading
import time
from time import gmtime, strftime

import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import widgets
from IPython.display import display
import tensorflow as tf

import gym
from gym import wrappers
import random

from matplotlib import animation
from JSAnimation.IPython_display import display_animation


class QLearningNetwork:

    def __init__(self):
        self._init_network_input()
        self._init_weights()
        self._init_biases()
        self._init_network_layout()

    def _init_network_input(self):
        self.network_state = tf.compat.v1.placeholder(tf.float32, [None, 4], name='input')
        self.network_action = tf.compat.v1.placeholder(tf.int32, [None], name='actioninput')
        self.network_reward = tf.compat.v1.placeholder(tf.float32, [None], name='groundtruth_reward')
        self.action_one_hot = tf.one_hot(self.network_action, 2, name='actiononehot')

    def _init_weights(self):
        self.w1 = tf.Variable(tf.random.normal([4, 16], stddev=0.35), name='w1')
        self.w2 = tf.Variable(tf.random.normal([16, 32], stddev=0.35), name='w2')
        self.w3 = tf.Variable(tf.random.normal([32, 8], stddev=0.35), name='w3')
        self.w4 = tf.Variable(tf.random.normal([8, 2], stddev=0.35), name='w4')

    def _init_biases(self):
        self.b1 = tf.Variable(tf.zeros([16]), name='B1')
        self.b2 = tf.Variable(tf.zeros([32]), name='B2')
        self.b3 = tf.Variable(tf.zeros([8]), name='B3')
        self.b4 = tf.Variable(tf.zeros([2]), name='B4')

    def _init_network_layout(self) -> None:
        self.layer_1 = tf.nn.relu(tf.add(tf.matmul(self.network_state, self.w1), self.b1), name='Result1')
        self.layer_2 = tf.nn.relu(tf.add(tf.matmul(self.layer_1, self.w2), self.b2), name='Result2')
        self.layer_3 = tf.nn.relu(tf.add(tf.matmul(self.layer_2, self.w3), self.b3), name='Result2')
        self.predicted_reward = tf.nn.relu(tf.add(tf.matmul(self.layer_3, self.w4), self.b4), name='predictedReward')

    def start_q_learning(self):
        global observation
        q_reward = tf.reduce_sum(tf.multiply(self.predicted_reward, self.action_one_hot), reduction_indices=1)
        loss = tf.reduce_mean(tf.square(self.network_reward - q_reward))
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.RMSPropOptimizer(0.0001).minimize(loss)
        merged_summary = tf.summary.merge_all()

        session = tf.InteractiveSession()
        summary_writer = tf.summary.FileWriter('train_summary', session.graph)
        session.run(tf.global_variables_initializer())

        replay_memory: list = []  # (state, action, reward, terminalstate, state_t+1)
        epsilon: float = 1.0
        BATCH_SIZE: int = 32
        GAMMA: float = 0.9
        MAX_LEN_REPLAY_MEMORY: int = 30000
        FRAMES_TO_PLAY: int = 300001
        MIN_FRAMES_FOR_LEARNING: int = 1000
        summary = None

        for i_epoch in range(FRAMES_TO_PLAY):

            # Select an action and perform this
            # EXERCISE: this is where your network should play and try to come as far as possible
            # You have to implement epsilon-annealing yourself
            action = env.action_space.sample()
            new_observation, reward, terminal, info = env.step(action)

            if terminal:  # agent gets 0 reward if it dies
                reward = 0

            # Add the observation to our replay memory
            replay_memory.append((observation, action, reward, terminal, new_observation))

            # Reset the environment if the agent died
            if terminal:
                new_observation = env.reset()
            observation = new_observation

            # Learn once we have enough frames to start learning
            if len(replay_memory) > MIN_FRAMES_FOR_LEARNING:
                experiences = random.sample(replay_memory, BATCH_SIZE)
                to_train: list = []  # (state, action, delayed_reward)

                # Calculate the predicted reward
                next_states = [var[4] for var in experiences]
                pred_reward = session.run(self.predicted_reward, feed_dict={self.network_state: next_states})

                # Set the 'ground truth': the value our network has to predict:
                for index in range(BATCH_SIZE):
                    state, action, reward, terminal_state, new_state = experiences[index]
                    self.predicted_reward = max(pred_reward[index])

                    if terminal_state:
                        delayed_reward = reward
                    else:
                        delayed_reward = reward + GAMMA * self.predicted_reward
                    to_train.append((state, action, delayed_reward))

                # Feed the train batch to the algorithm
                states: list = [elem[0] for elem in to_train]
                actions: list = [elem[1] for elem in to_train]
                rewards: list = [elem[2] for elem in to_train]

                _, l, summary = session.run([optimizer, loss, merged_summary],
                                            feed_dict={self.network_state: states, self.network_action: actions,
                                                       self.network_reward: rewards})

                # If our memory is too big: remove the first element
                if len(replay_memory) > MAX_LEN_REPLAY_MEMORY:
                    replay_memory = replay_memory[1:]

                # Show the progress
                if i_epoch % 100 == 1:
                    summary_writer.add_summary(summary, i_epoch)
                if i_epoch % 1000 == 1:
                    print('Epoch: {}, loss: {}'.format(i_epoch, l))

            # Play until we are dead
            observation = env.reset()
            term: bool = False
            predicted_q: list = []
            frames: list = []

            while not term:
                rgb_observation = env.render(mode='rgb_array')
                frames.append(rgb_observation)
                pred_q = session.run(self.predicted_reward, feed_dict={self.network_state: [observation]})
                predicted_q.append(pred_q)
                action = np.argmax(pred_q)
                observation, _, term, _ = env.step(action)

            # Plot the replay
            self.display_frames_as_gif(frames, filename_gif='dqn_run.gif')

    def display_frames_as_gif(self, frames: list, filename_gif: str = None) -> None:
        """
        Displays a list of frames as a gif, with controls
        :param frames: the list of frames
        :param filename_gif: if given, the gif will be stored with this filename
        :return: None
        """
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        if filename_gif:
            anim.save(filename_gif, writer='imagemagick', fps=20)
        display(display_animation(anim, default_mode='loop'))


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation = env.reset()
    network = QLearningNetwork()
