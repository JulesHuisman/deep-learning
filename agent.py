import random
import math
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

class Trader:
    def __init__(self,
                 state_size,
                 discount_rate=0.99,
                 action_size=3,
                 neurons=128,
                 neuron_shrink_ratio=.5,
                 hidden_layers=3,
                 learning_rate=0.00025,
                 sample_batch_size=64):

        self.discount_rate       = discount_rate
        self.state_size          = state_size
        self.action_size         = action_size
        self.sample_batch_size   = sample_batch_size
        self.memory              = deque(maxlen=sample_batch_size*action_size)
        self.learning_rate       = learning_rate
        self.neurons             = neurons
        self.neuron_shrink_ratio = neuron_shrink_ratio
        self.hidden_layers       = hidden_layers

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        neurons = self.neurons

        model.add(Dense(neurons, input_dim=self.state_size, activation='relu'))

        for hidden_layer in range(self.hidden_layers - 1):
            neurons *= self.neuron_shrink_ratio
            model.add(Dense(math.ceil(neurons), activation='relu'))

        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def reset_model(self):
        def reset_weights(model):
            session = K.get_session()

            for layer in self.model.layers: 
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel.initializer.run(session=session)

    def get_action(self, state):
        # return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def memory_filled(self):
        return len(self.memory) >= self.memory.maxlen

    def replay(self):
        # Don't replay if the memory is not filled yet
        if not self.memory_filled():
            return

        for (state, action, reward, next_state) in self.memory:
            # If there is no next state
            if len(next_state) == 0:
                continue

            model_pred = self.model.predict(next_state)

            target = reward + (self.discount_rate * np.amax(model_pred[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
