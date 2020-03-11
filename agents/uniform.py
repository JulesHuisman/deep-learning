from agents.agent import Agent
from utils import *
from collections import deque

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from scipy.special import softmax

import numpy as np
import random

class Uniform(Agent):
    def __init__(self):
        super(Uniform, self).__init__()

    def _model(self, window_size, action_size):
        return None

    def act(self, state, position):
        """
        Take a new position
        """
        uniform = np.array([1 / self.action_size] * self.action_size)

        q_value = uniform
        position = uniform

        return position, q_value

    def train(self, done):
        pass