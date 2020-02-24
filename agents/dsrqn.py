from agents.agent import Agent

from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Input, Lambda, Reshape, Flatten, MaxPooling1D, AveragePooling1D, LSTM, LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from scipy.special import softmax

import numpy as np
import random

class DSRQN(Agent):
    def __init__(self):
        super(DSRQN, self).__init__()

    def _model(self, action_size):
        """
        Create a keras model to learn the optimal porfolio policy
        """
        X = Input(shape=(None, (action_size-1)))

        lstm = LSTM(32, activation='relu', kernel_initializer='he_uniform')(X)
        fc = Dense(24, activation='relu', kernel_initializer='he_uniform')(lstm)

        y = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(fc)

        model = Model(X, y)

        model.compile(loss='mse', optimizer=Adam(lr=self.lr))

        return model

    def act(self, state):
        """
        Take an action based on a certain state
        """
        action = self.model.predict(state)[0]
        position = softmax(action)

        return action, position

    def train(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        target_val = self.model.predict(next_state)[0]

        # print('target', target)
        # print('action', action)
        # print('reward', reward)
        # print('target_val', target_val)
        # print('done', done)
        reward *= 100

        if done:
            target[0][np.argmax(action)] = reward
        else:
            target[0][np.argmax(action)] = reward + self.gamma * np.amax(target_val)

        # print('target with reward', target)
        # print()

        self.model.fit(state, target, epochs=1, verbose=0, batch_size=1)

        # target = self.model.predict(update_input)
        # target_val = self.target_model.predict(update_target)

        # for i in range(batch_size):
        #     if done[i]:
        #         target[i][action[i]] = reward[i]
        #     else:
        #         target[i][action[i]] = reward[i] + \
        #             self.gamma * np.amax(target_val[i])

        # self.model.fit(update_input, target,
        #                batch_size=batch_size, epochs=1, verbose=0)