import random
import numpy as np

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Trader:
    def __init__(self,
                state_size,
                discount_rate=0.85,
                action_size=3,
                learning_rate=0.0001,
                sample_batch_size=64):

        self.discount_rate = discount_rate
        self.state_size = state_size
        self.action_size = action_size
        self.sample_batch_size = sample_batch_size
        self.memory = deque(maxlen=sample_batch_size*action_size)
        self.learning_rate = learning_rate

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def get_action(self, state):
        return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self):
        # Don't replay if the memory is not filled yet
        if len(self.memory) < self.memory.maxlen:
            return

        # sample_batch = random.sample(self.memory, self.sample_batch_size)

        for (state, action, reward, next_state) in self.memory:
            # If there is no next state
            if len(next_state) == 0:
                continue

            model_pred = self.model.predict(next_state)

            target = reward + (self.discount_rate * np.amax(model_pred[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
