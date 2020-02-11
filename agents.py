import random
import numpy as np

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQN:
    def __init__(self,
                state_size,
                discount_rate=0.95,
                exploration_rate=1.,
                exploration_decay=.995,
                action_size=3,
                learning_rate=0.01,
                is_eval=False):

        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.learning_rate = learning_rate
        self.is_eval = is_eval

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        model.add(Dense(32, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def get_next_action(self, state):
        if (random.random() > self.exploration_rate) or self.is_eval:
            return self.model_action(state)
        else:
            return self.random_action()

    def model_action(self, state):
        return np.argmax(self.model.predict(state)[0])

    def random_action(self):
        return random.randrange(self.action_size)

    def remember(self, state, next_state, action, reward):
        self.memory.append((state, next_state, action, reward))

    def replay(self, sample_batch_size):
        if self.is_eval:
            return

        if len(self.memory) < sample_batch_size:
            return

        sample_batch = random.sample(self.memory, sample_batch_size)

        for (state, next_state, action, reward) in sample_batch:
            # If there is no next state
            if len(next_state) == 0:
                continue

            model_pred = self.model.predict(next_state)

            target = reward + (self.discount_rate * np.amax(model_pred[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.train(state, target_f)

        self.update_rate()

    def train(self, x, y):
        self.model.fit(x, y, epochs=1, verbose=0)

    def update_rate(self):
        # Keep lowering the exploration rate
        self.exploration_rate *= self.exploration_decay
