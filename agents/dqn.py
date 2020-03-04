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

MEMORY_SIZE = 3_000
MINIBATCH_SIZE = 64

class DQN(Agent):
    def __init__(self):
        super(DQN, self).__init__()

        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = .9995
        self.train_start   = 1_000
        self.lr = 0.001

        self.memory = deque(maxlen=MEMORY_SIZE)

    def setup(self, **kwargs):
        # Call the setup of the base agent
        super(DQN, self).setup(**kwargs)

        # Also add a target model
        self.target_model = self._model(window_size=kwargs['window_size'],
                                        action_size=kwargs['action_size'])

        # Copy the weights
        self.update_target_model()

    def update_target_model(self):
        """
        Copy the weights of the orignal model to the target model
        """
        self.target_model.set_weights(self.model.get_weights())

    def _model(self, window_size, action_size):
        """
        Create a keras model to learn the optimal porfolio policy
        """
        # Two inputs, historic log returns and current position of the agent
        historic_input = Input(shape=(window_size, (action_size - 1)), name='historic_input')
        position_input = Input(shape=(action_size,), name='position_input')

        # # Flatten the historic input because no RNN is used
        # flat = Flatten()(historic_input)
        lstm = LSTM(32, activation='relu', name='lstm')(historic_input)
        # norm = BatchNormalization(name='norm-1')(lstm)

        # Join the historic and position flows
        x = concatenate([lstm, position_input], name='merge')

        fc = Dense(32, activation='relu', name='dense')(x)
        # norm = BatchNormalization(name='norm-2')(fc)

        # Output layer
        y = Dense(action_size, activation='linear', name='output')(fc)

        model = Model(inputs=[historic_input, position_input], outputs=y)

        model.compile(loss='mse', optimizer=Adam(lr=self.lr))

        return model

    def act(self, state, position):
        """
        Take a new position based on the model of the agent
        # https://github.com/edwardhdlu/q-trader/blob/master/agent/agent.py
        """
        q_value = self.model.predict([np.array([state]), np.array([position])], batch_size=1)[0]

        # Random action
        if (np.random.rand() < self.epsilon) and self.training:
            position = one_hot(self.action_size, np.random.randint(self.action_size))
        # Greedy action
        else:
            position = one_hot(self.action_size, np.argmax(q_value))

        return position, q_value

    def remember(self, state, position, reward, next_state, next_position, done):
        """
        Store information in the memory of the agent. Used for experience replay
        """
        if self.training:
            self.memory.append((state, position, reward, next_state, next_position, done))

    def train(self, done):
        """
        Train the agent
        # https://www.youtube.com/watch?v=qfovbG84EBg
        """
        # Agent must be training
        if not self.training:
            return

        # Wait until threshold to start training
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, MINIBATCH_SIZE)

        states         = np.array([transition[0] for transition in minibatch])
        positions      = np.array([transition[1] for transition in minibatch])
        q_values       = self.model.predict([states, positions])
        next_states    = np.array([transition[3] for transition in minibatch])
        next_positions = np.array([transition[4] for transition in minibatch])
        next_q_values  = self.target_model.predict([next_states, next_positions])

        X_hist, X_pos, y = [], [], []

        for index, (state, position, reward, next_state, next_position, done) in enumerate(minibatch):
            if not done:
                max_next_q = np.max(next_q_values[index])
                target_q = reward + self.gamma * max_next_q
            else:
                target_q = reward

            q_value = q_values[index]
            q_value[np.argmax(position)] = target_q

            X_hist.append(state)
            X_pos.append(position)
            y.append(q_value)

        # Transform to numpy arrays
        X_hist, X_pos, y = np.array(X_hist), np.array(X_pos), np.array(y)

        # Fit the model
        self.model.fit([X_hist, X_pos], y, batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update the target model at the end of the episode
        if done:
            self.update_target_model()