from utils import *
from collections import deque
from keras.utils import plot_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Agent:
    def __init__(self, training=True):
        # Create a dataframe to store the actions taken
        self.positions = {}
        self.rewards   = {}
        self.returns   = {}
        self.states    = {}
        self.q_values  = {}

        self.gamma = 0.99
        self.lr    = 0.01

        # Window and action size
        self.window_size = None
        self.action_size = None

        # Model
        self.model = None

        # Is training
        self.training = None

        # Portfolio value
        self.portfolio_value = 1

    @property
    def mode(self):
        """Training or testing mode"""
        return 'training' if self.training else 'testing'

    def _model(self, action_size):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def print_model(self):
        if self.model:
            return plot_model(self.model, show_shapes=True, dpi=64)

    def setup(self, index, columns, window_size, action_size, training):
        """
        Setup the agent
        """
        self.training = training

        self.window_size = window_size
        self.action_size = action_size

        if not self.model:
            self.model = self._model(window_size, action_size)

        # Create dataframes to record positions and rewards
        positions = df_create(index=index, columns=columns)
        rewards   = df_create(index=index, columns=['rewards'])
        returns   = df_create(index=index, columns=['returns'])
        states    = df_create(index=index, columns=['state'])
        q_values  = df_create(index=index, columns=columns)

        self.positions[self.mode] = positions
        self.rewards[self.mode]   = rewards
        self.returns[self.mode]   = returns
        self.states[self.mode]    = states
        self.q_values[self.mode]  = q_values

    def reset(self):
        """
        Reset the agent
        """
        # Set the portfolio value back to one
        self.portfolio_value = 1

        # Reset the logging dataframes
        self.positions[self.mode] = df_clean(self.positions[self.mode])
        self.rewards[self.mode]   = df_clean(self.rewards[self.mode])
        self.returns[self.mode]   = df_clean(self.returns[self.mode])
        self.states[self.mode]    = df_clean(self.states[self.mode])
        self.q_values[self.mode]  = df_clean(self.q_values[self.mode])

        # Allows to store numpy arrays in cells
        self.states[self.mode]['state'] = self.states[self.mode]['state'].astype(object)

    def save_logs(self, episode):
        """Store the progress dataframe"""
        self.positions[self.mode].iloc[self.window_size:].to_pickle(f'data/logs/{self.mode}-positions-{episode}.pkl')
        self.rewards[self.mode].iloc[self.window_size:].to_pickle(f'data/logs/{self.mode}-rewards-{episode}.pkl')
        self.returns[self.mode].iloc[self.window_size:].to_pickle(f'data/logs/{self.mode}-returns-{episode}.pkl')
        self.q_values[self.mode].iloc[self.window_size:].to_pickle(f'data/logs/{self.mode}-q-values-{episode}.pkl')

    @staticmethod
    def load_logs(episode, mode):
        positions_file = f'data/logs/{mode}-positions-{episode}.pkl'
        rewards_file = f'data/logs/{mode}-rewards-{episode}.pkl'
        returns_file = f'data/logs/{mode}-returns-{episode}.pkl'
        q_values_file = f'data/logs/{mode}-q-values-{episode}.pkl'

        # Make sure files exist
        assert os.path.isfile(positions_file) and os.path.isfile(rewards_file) and os.path.isfile(returns_file) and os.path.isfile(q_values_file), 'Log files do not exist'

        # Load the files
        positions = pd.read_pickle(positions_file)
        rewards   = pd.read_pickle(rewards_file)
        returns   = pd.read_pickle(returns_file)
        q_values  = pd.read_pickle(q_values_file)

        return positions, rewards, returns, q_values

    @property
    def summary(self):
        """
        Create a summary of the current episode
        """

        # Differences between subsequent positions
        pos_diff = np.diff(np.argmax(self.positions[self.mode].values, 1))

        return {
            'mean_returns': np.mean(self.returns[self.mode]).iloc[0],
            'cumulative_returns': cum_returns(self.returns[self.mode]).iloc[-1, 0],
            'sharpe_ratio': sharpe_ratio(self.returns[self.mode]).iloc[0],
            'maximum_drawdown': max_drawdown(self.returns[self.mode]).min().iloc[0],
            'exploration_rate': self.epsilon if hasattr(self, 'epsilon') else 0,
            'position_changes': len(pos_diff[pos_diff != 0]) / len(self.positions[self.mode].iloc[self.window_size:-1])
        }