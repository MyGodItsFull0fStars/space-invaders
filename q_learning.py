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


class Network:

    def __init__(self):
        self._init_network_input()
        self._init_weights()
        self._init_biases()
        self._init_network_layout()

    def _init_network_input(self):
        self.network_state = tf.placeholder(tf.float32, [None, 4], name='input')
        self.network_action = tf.placeholder(tf.int32, [None], name='actioninput')
        self.network_reward = tf.placeholder(tf.float32, [None], name='groundtruth_reward')
        self.action_one_hot = tf.one_hot(self.network_action, 2, name='actiononehot')

    def _init_weights(self):
        self.w1 = tf.Variable(tf.random_normal([4, 16], stddev=0.35), name='w1')
        self.w2 = tf.Variable(tf.random_normal([16, 32], stddev=0.35), name='w2')
        self.w3 = tf.Variable(tf.random_normal([32, 8], stddev=0.35), name='w3')
        self.w4 = tf.Variable(tf.random_normal([8, 2], stddev=0.35), name='w4')

    def _init_biases(self):
        self.b1 = tf.Variable(tf.zeros([16]), name='B1')
        self.b2 = tf.Variable(tf.zeros([32]), name='B2')
        self.b3 = tf.Variable(tf.zeros([8]), name='B3')
        self.b4 = tf.Variable(tf.zeros([2]), name='B4')

    def _init_network_layout(self) -> None:
        self.layer_1 = tf.nn.relu(tf.add(tf.matmul(self.network_state, self.w1), self.b1), name='Result 1')
        self.layer_2 = tf.nn.relu(tf.add(tf.matmul(self.layer_1, self.w2), self.b2), name='Result 2')
        self.layer_3 = tf.nn.relu(tf.add(tf.matmul(self.layer_2, self.w3), self.b3), name='Result 2')
        self.predicted_reward = tf.nn.relu(tf.add(tf.matmul(self.layer_3, self.w4), self.b4), name='Predicted Reward')

    def start_q_learning(self):
        q_reward = tf.reduce_sum(tf.multiply(self.predicted_reward, self.action_one_hot), reduction_indices=1)
        loss = tf.reduce_mean(tf.square(self.network_reward - q_reward))
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.RMSPropOptimizer(0.0001).minimize(loss)
        merged_summary = tf.summary.merge_all()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation = env.reset()
