from agents.agent import Agent
from utils import *

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate
from keras.optimizers import Adam
from keras import backend as K

import numpy as np
import random

class REINFORCE(Agent):
    def __init__(self):
        super(REINFORCE, self).__init__()

        self.exploration_rate = 1.0
        self.exploration_decay = -0.03
        self.lr = 0.001

    def _model(self, action_size):
        """
        Create a keras model to learn the optimal porfolio policy
        """
        historic_input = Input(shape=(None, (action_size-1)), name='historic_input')
        position_input = Input(shape=(action_size,), name='position_input')

        lstm = LSTM(32, activation='relu', name='lstm')(historic_input)

        x = concatenate([lstm, position_input], name='merge')

        fc = Dense(32, activation='relu', name='dense')(x)

        y = Dense(action_size, activation='softmax', name='output')(fc)

        model = Model(inputs=[historic_input, position_input], outputs=y)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))

        return model

    def _discount_rewards(self, rewards):
        """
        Takes 1d float array of rewards and computes discounted reward
        e.g. f([1, 1, 1, 1], 0.99) -> [3.94, 2.97, 1.99, 1.0]
        """
        prior = 0
        out = []

        for val in rewards:
            new_val = val + prior * self.gamma
            out.append(new_val)
            prior = new_val 
            
        return np.array(out[::-1])

    def action_to_position(self, action):
        return one_hot(self.action_size, action)

    def act(self, state, position):
        policy = self.model.predict([state, np.array([position])], batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return self.action_to_position(action)

    def train(self):
        rewards   = self.rewards.values[30:-1, 0]
        # rewards   = cum_returns(self.returns.values[30:-1, 0])
        states    = self.states.values[30:-1, 0]
        positions = self.positions.values[30:-1]

        actions = np.argmax(positions, 1)

        episode_length = len(states)

        discounted_rewards = self._discount_rewards(rewards)
        discounted_rewards = normalize(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.window_size, self.action_size-1))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = states[i]
            advantages[i][actions[i]] = discounted_rewards[i]

        self.discounted_rewards = discounted_rewards
        self.update_inputs = update_inputs
        self.advantages = advantages

        self.model.fit([update_inputs, positions], advantages, epochs=1, batch_size=episode_length, verbose=0)