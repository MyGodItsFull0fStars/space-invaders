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

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation = env.reset()
