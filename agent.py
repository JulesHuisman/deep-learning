import random
import math
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Input, Lambda, Reshape, Flatten, MaxPooling1D, AveragePooling1D, LSTM
from keras.optimizers import Adam, RMSprop, SGD
from keras.initializers import RandomNormal
from keras import backend as K
from utils import clone_model

class Trader:
    def __init__(self,
                 state_size,
                 discount_rate=0.85,
                 action_size=3,
                 neurons=100,
                 neuron_shrink_ratio=.5,
                 hidden_layers=2,
                 learning_rate=0.00025,
                 sample_batch_size=64,
                 target_tau=0.125,
                 memory_size=365):

        self.discount_rate       = discount_rate
        self.state_size          = state_size
        self.action_size         = action_size
        self.sample_batch_size   = sample_batch_size
        self.memory              = deque(maxlen=sample_batch_size*action_size)
        self.learning_rate       = learning_rate
        self.neurons             = neurons
        self.neuron_shrink_ratio = neuron_shrink_ratio
        self.hidden_layers       = hidden_layers
        self.target_tau          = target_tau

        self.model = self.build_model()
        self.target_model = clone_model(self.model)

    def build_model(self):
        model = Sequential()

        neurons = self.neurons

        # model.add(Reshape((-1, 1), input_shape=(self.state_size,)))
        # model.add(MaxPooling1D(4, strides=1))
        # model.add(AveragePooling1D(7, strides=1))
        # model.add(Conv1D(12, kernel_size=7, activation='relu'))
        # model.add(AveragePooling1D(180, strides=1))
        # model.add(Flatten())

        model.add(Dense(neurons, 
                        input_shape=(self.state_size,), 
                        activation='relu'))

        # model.add(Dense(64, 
        #                 activation='relu',
        #                 kernel_initializer=RandomNormal(stddev=0.001),
        #                 bias_initializer='zeros'))

        # model.add(LSTM(64, activation='tanh'))
        
        for hidden_layer in range(self.hidden_layers - 1):
            neurons *= self.neuron_shrink_ratio
            model.add(Dense(math.ceil(neurons), activation='relu'))

        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='huber_loss', optimizer=Adam(lr=self.learning_rate))
        # model.compile(loss='huber_loss', optimizer='rmsprop')

        return model

    def reset_model(self):
        """
        Reset both the model and target model
        """
        self.model = self.build_model()
        self.target_model = clone_model(self.model)

    def hard_update_target_model(self):
        """
        Copy the weights of the model to the target model
        """
        self.target_model.set_weights(self.model.get_weights())

    def soft_update_target_model(self):
        """
        Slowly move the weights of the target model towards the weight of the trained model.
        https://github.com/javimontero/deep-q-trader
        """
        weights        = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = (self.target_tau * weights[i]) + ((1 - self.target_tau) * target_weights[i])

        self.target_model.set_weights(target_weights)

    def get_action(self, state):
        """
        Get a action from the model
        """
        # return np.random.randint(0,3)
        # print(state)
        # state = np.random.randn(*state.shape)
        # print(state)
        # print(self.model.predict(state))
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, next_state):
        """
        Store states, actions and rewards in the memory of the trader
        """
        if len(next_state) > 0:
            self.memory.append((np.array(state), action, reward, np.array(next_state)))

    def memory_filled(self):
        """
        Check if the memory is filled up
        """
        return len(self.memory) >= self.memory.maxlen

    def replay(self):
        """
        Memory replay, this is where the trades learns optimal values for each market state
        """
        # Don't replay if the memory is not filled yet
        if not self.memory_filled():
            return

        # Transform the historic data into numpy matrices
        states      = np.array([memory[0][0] for memory in self.memory])
        actions     = np.array([memory[1] for memory in self.memory])
        rewards     = np.array([memory[2] for memory in self.memory])
        next_states = np.array([memory[3][0] for memory in self.memory])
        # states      = np.array([memory[0][0] for memory in self.memory])[:-3,:]
        # actions     = np.array([memory[1] for memory in self.memory])[:-3]
        # rewards     = np.array([memory[2] for memory in self.memory])[:-3]
        # next_states = np.array([memory[3][0] for memory in self.memory])[:-3,:]

        # print(states[:-3,:].shape, actions[:-3].shape, rewards[:-3].shape, next_states[:-3,:].shape)

        # Predict the actions from the future states
        predictions = self.target_model.predict(next_states)

        # Bellman equation
        target = rewards + np.amax(predictions, 1) * self.discount_rate

        # Create the matrix to train the Q network model on
        target_f = np.copy(predictions)
        np.put_along_axis(target_f, np.expand_dims(actions, axis=1), np.expand_dims(target, axis=1), axis=1)

        # states = np.random.randn(*states.shape) * 10

        # print(states[:6,:5])

        # Train the model
        self.model.fit(states, 
                       target_f, 
                       epochs=1, 
                       verbose=0,
                       batch_size=16)
