import pandas as pd

from pandas.tseries.offsets import BDay
from utils import *

class Env:
    def __init__(self, stocks, window_size=30):
        self._prices = stocks.prices

        # Add cash to be able to trade to cash
        self._prices['CASH'] = 1.0

        # Percentage wise returns
        self._returns = simple_returns(self._prices)

        # Percentage wise returns
        self._log_returns = log_returns(self._prices)

        # Window size
        self.window_size = window_size

        # Action size
        self.action_size = len(self.universe)

        # Counter to keep track of progress
        self._counter = 0

    @property
    def prices(self):
        """Return the stock prices"""
        return self._prices

    @property
    def returns(self):
        """Simple returns of the assets"""
        return self._returns

    @property
    def index(self):
        """Current day of the environment"""
        return self._prices.index[self._counter]

    @property
    def universe(self):
        """Constituents of the portfolio"""
        return self._prices.columns

    @property
    def dates(self):
        """All the different dates in the environment"""
        return self._prices.index

    @property
    def number_of_steps(self):
        """The total number of steps you can take in the environment"""
        return len(self.dates)

    @property
    def state(self):
        """Get the current state of the environment"""
        # return self._prices.loc[self.index]
        return self._get_window(start=(self._counter-self.window_size), end=self._counter)

    @property
    def next_state(self):
        """Get the state of the next business day of the environment"""
        # return self._prices.loc[self.index + BDay(1)]
        return self._get_window(start=(self._counter-self.window_size+1), end=(self._counter+1))

    def _get_window(self, start, end):
        """Get a window of the log returns"""
        # Get prices without cash
        prices = (self._prices
                    .drop(['CASH'], axis=1, errors='ignore'))

        # Get the logaritmic returns
        lr = (log_returns(prices)
                    .iloc[max(start, 0):end]
                    .values)

        return lr.reshape(1, -1, (self.action_size-1))

    def _is_done(self):
        return (self.index == self.dates[-1])

    def _get_reward(self, action):
        return self._returns.loc[self.index] * action

    def register(self, agent):
        """
        Register an agent to this environment
        """
        # Setup the agent to fit to the environment
        agent.setup(index=self.dates, 
                    columns=self.universe,
                    action_size=self.action_size)

        # Register the agent
        self.agent = agent

    def reset(self):
        """Reset the environment"""
        self._counter = 1

        # Reset the agent
        self.agent.reset()

        # Return the first state
        return self.state

    def step(self, action, position):
        """
        Let the agent take one step in the environment
        """
        # Observe the current environment
        state      = self.state
        next_state = self.next_state
        done       = self._is_done()

        # print(self._counter, position)

        # Get the reward of the action
        reward = self._get_reward(position).sum()

        # Store action and reward for agent
        self.agent.positions.loc[self.index] = position
        self.agent.rewards.loc[self.index] = reward

        self.agent.train(state, action, reward, next_state, done)

        # Store in agent memory
        # self.agent.remember(state, action, reward, next_state, done)

        # Increase the counter
        self._counter += 1

        return next_state, reward, done

