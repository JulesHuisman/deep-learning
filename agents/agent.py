import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *
from collections import deque

class Agent:
    def __init__(self, memory=10000):
        # Create a dataframe to store the actions taken
        self.positions = None
        self.rewards = None

        self.gamma = 0.99
        self.lr = 10

        # State and action size
        self.state_size = None
        self.action_size = None

        # Model
        self.model = None

        # Memory
        self.memory = deque(maxlen=memory)

    def _model(self, action_size):
        raise NotImplementedError

    def setup(self, index, columns, action_size):
        """
        Setup the agent
        """
        self.action_size = action_size

        self.model = self._model(action_size)

        # Create dataframes to record positions and rewards
        self._positions = df_create(index=index, columns=columns)
        self._rewards   = df_create(index=index, columns=['Rewards'])

    def remember(self, state, position, reward, next_state, done):
        self.memory.append((state, position, reward, next_state, done))

    def act(self, state):
        raise NotImplementedError

    def reset(self):
        """Reset the logging dataframes"""
        self.positions = df_clean(self._positions)
        self.rewards   = df_clean(self._rewards)

    def save_logs(self, episode):
        """Store the progress dataframe"""
        self.positions.to_pickle(f'data/logs/positions-{episode}.pkl')
        self.rewards.to_pickle(f'data/logs/rewards-{episode}.pkl')

    @staticmethod
    def load_logs(episode):
        position_file = f'data/logs/positions-{episode}.pkl'
        rewards_file = f'data/logs/rewards-{episode}.pkl'

        # Make sure files exist
        assert os.path.isfile(position_file) and os.path.isfile(rewards_file), 'Log files do not exist'

        # Load the files
        positions = pd.read_pickle(position_file)
        rewards   = pd.read_pickle(rewards_file)

        return positions, rewards
