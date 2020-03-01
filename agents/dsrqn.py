from agents.agent import Agent

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate
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
        historic_input = Input(shape=(None, (action_size-1)), name='historic_input')
        position_input = Input(shape=(action_size,), name='position_input')

        lstm = LSTM(32, activation='relu', name='lstm')(historic_input)

        x = concatenate([lstm, position_input], name='merge')

        fc = Dense(32, activation='relu', name='dense')(x)

        y = Dense(action_size, activation='linear', name='output')(fc)

        model = Model(inputs=[historic_input, position_input], outputs=y)

        model.compile(loss='mse', optimizer=Adam(lr=self.lr))

        return model

    def q_value(self, state):
        """Get the q value of a certain state"""
        return self.model.predict([state, np.array([self.position])])[0]

    def train(self, state, position, reward, next_state, done):
        pass
        # target = self.model.predict(state)
        # target_val = self.model.predict(next_state)

        # # print('target', target)
        # # print('action', action)
        # # print('reward', reward)
        # # print('target_val', target_val)
        # # print('done', done)

        # if done:
        #     target[0][action] = reward
        # else:
        #     target[0][action] = reward + self.gamma * np.amax(target_val)

        # # print('target with reward', target)
        # # print()

        # self.model.fit(state, target, epochs=1, verbose=0, batch_size=1)

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