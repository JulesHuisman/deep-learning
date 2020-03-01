import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *
from collections import deque

class Agent:
    def __init__(self):
        # Create a dataframe to store the actions taken
        self.positions = None
        self.rewards   = None
        self.returns   = None
        self.states    = None

        self.gamma = 0.99
        self.lr    = 0.001

        # Window and action size
        self.window_size = None
        self.action_size = None

        # Model
        self.model = None

    def _model(self, action_size):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def setup(self, index, columns, window_size, action_size):
        """
        Setup the agent
        """
        self.window_size = window_size
        self.action_size = action_size

        self.model = self._model(action_size)

        # Create dataframes to record positions and rewards
        self.positions = df_create(index=index, columns=columns)
        self.rewards   = df_create(index=index, columns=['rewards'])
        self.returns   = df_create(index=index, columns=['returns'])
        self.states    = df_create(index=index, columns=['state'])

    def reset(self):
        """Reset the logging dataframes"""
        self.positions = df_clean(self.positions)
        self.rewards   = df_clean(self.rewards)
        self.returns   = df_clean(self.returns)
        self.states    = df_clean(self.states)

        # Allows to store numpy arrays in cells
        self.states['state'] = self.states['state'].astype(object)

    def save_logs(self, episode):
        """Store the progress dataframe"""
        self.positions.to_pickle(f'data/logs/positions-{episode}.pkl')
        self.rewards.to_pickle(f'data/logs/rewards-{episode}.pkl')
        self.returns.to_pickle(f'data/logs/returns-{episode}.pkl')

    @staticmethod
    def load_logs(episode):
        positions_file = f'data/logs/positions-{episode}.pkl'
        rewards_file = f'data/logs/rewards-{episode}.pkl'
        returns_file = f'data/logs/returns-{episode}.pkl'

        # Make sure files exist
        assert os.path.isfile(positions_file) and os.path.isfile(rewards_file) and os.path.isfile(returns_file), 'Log files do not exist'

        # Load the files
        positions = pd.read_pickle(positions_file)
        rewards   = pd.read_pickle(rewards_file)
        returns  = pd.read_pickle(returns_file)

        return positions, rewards, returns

    @property
    def summary(self):
        """
        Create a summary of the current episode
        """

        return {
            'Mean returns': np.mean(self.rewards).iloc[0],
            'Cumulative returns': cum_returns(self.rewards).iloc[-1,0],
            'Sharpe ratio': sharpe_ratio(self.rewards).iloc[0],
            'Maximum drawdown': max_drawdown(self.rewards).min().iloc[0],
            'Exploration rate': self.exploration_rate if hasattr(self, 'exploration_rate') else 0
        }

    def print_summary(self):
        """Print a summary of the current episode"""
        for key, value in self.summary.items():
            print(key, value)