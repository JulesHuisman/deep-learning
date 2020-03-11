import pandas as pd

from utils import *
import math

class Env:
    def __init__(self, stocks, logger, window_size=30, fee=0.002):
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
        self.counter = 0

        # Trading fee
        self.fee = fee

        # Tensorboard logger
        self.logger = logger

    @property
    def prices(self):
        """Return the stock prices"""
        return self._prices

    @property
    def returns(self):
        """Simple returns of the assets"""
        return self._returns

    @property
    def log_returns(self):
        """Logaritmic returns of the assets"""
        return log_returns(self._prices)

    @property
    def today(self):
        """Current day of the environment"""
        return self._prices.index[self.counter]

    @property
    def yesterday(self):
        """Previous day of the environment"""
        return self._prices.index[self.counter - 1]

    @property
    def tomorrow(self):
        """Next day of the environment"""
        return self._prices.index[self.counter + 1]

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
        return self._get_window(start=(self.counter - self.window_size), end=self.counter)

    @property
    def next_state(self):
        """Get the state of the next business day of the environment"""
        return self._get_window(start=(self.counter - self.window_size + 1), end=(self.counter + 1))

    def _get_window(self, start, end):
        """Get a window of the log returns"""
        # Get prices without cash
        prices = (self._prices
                    .drop(['CASH'], axis=1, errors='ignore'))

        # Get the logaritmic returns
        lr = (log_returns(prices)
                    .iloc[max(start, 0):end]
                    .values)

        return lr.reshape(-1, (self.action_size - 1))

    def _is_done(self):
        return (self.today == self.dates[-2])

    def _get_return(self, prev_position, position):
        """
        Get the return of the current position
        """
        delta = sum(abs(position[:-1] - prev_position[:-1]))
        return np.sum(self._returns.loc[self.today] * position) - (delta * self.fee)

    def _get_reward(self, prev_position, position):
        """
        Get the reward of holding the current position
        """
        previous_portfolio_value = self.agent.portfolio_value

        fee = sum(abs(position[:-1] - prev_position[:-1])) * previous_portfolio_value * self.fee

        # Growth ratio
        # Simple stock returns [0.2, 0, -0.2]
        # Position             [1,   0,  0  ]
        # Growth ratio         0.2 + 1 = 1.2
        growth_ratio = np.sum(self._returns.loc[self.today] * position) + 1

        new_portfolio_value = (previous_portfolio_value * growth_ratio) - fee

        # print('Position:', position)
        # print('Returns', self._returns.loc[self.today].values)
        # print('Portfolio value:', self.agent.portfolio_value)
        # print('Growth ratio', growth_ratio)
        # print('Fee', fee)
        # print('New portfolio value', new_portfolio_value)
        # print('Reward', math.log(new_portfolio_value / previous_portfolio_value))
        # print()

        # Update portfolio value
        self.agent.portfolio_value = new_portfolio_value

        return math.log(new_portfolio_value / previous_portfolio_value)

    def register(self, agent, training):
        """
        Register an agent to this environment
        """
        # Setup the agent to fit to the environment
        agent.setup(index=self.dates,
                    columns=self.universe,
                    window_size=self.window_size,
                    action_size=self.action_size,
                    training=training)

        # Register the agent
        self.agent = agent

    def reset(self):
        """Reset the environment"""
        self.counter = self.window_size

        # Reset the agent
        self.agent.reset()

        # Default position
        position = one_hot(self.action_size, self.action_size - 1)

        # Return the first state
        return self.state, position

    def step(self, next_position, q_value):
        """
        Let the agent take one step in the environment
        """
        # Observe the current environment
        state      = self.state
        next_state = self.next_state
        done       = self._is_done()

        prev_position = self.agent.positions[self.agent.mode].loc[self.yesterday].values
        position      = self.agent.positions[self.agent.mode].loc[self.today].values

        # Get the reward of the action
        returns = self._get_return(prev_position, position)
        reward = self._get_reward(prev_position, position)

        # Store position and reward for agent
        self.agent.positions[self.agent.mode].loc[self.tomorrow] = next_position
        self.agent.rewards[self.agent.mode].loc[self.today]      = reward
        self.agent.returns[self.agent.mode].loc[self.today]      = returns
        self.agent.q_values[self.agent.mode].loc[self.today]     = q_value

        # If the agent can remember
        if hasattr(self.agent, 'remember'):
            # self.agent.remember(state, position, reward, next_state, next_position, done)
            for i in range(self.action_size):
                position = one_hot(self.action_size, i)
                reward = self._get_reward(prev_position, position)

                self.agent.remember(state, position, reward, next_state, next_position, done)

        # Train the agent
        self.agent.train(done)

        # Increase the counter
        self.counter += 1

        return next_state, next_position, done

    def tensorboard(self, episode):
        """
        Create a tensorboard log
        """
        info = self.agent.summary

        self.logger.log_scalar('Cumulative returns', info['cumulative_returns'], episode)
        self.logger.log_scalar('Mean returns', info['mean_returns'], episode)
        self.logger.log_scalar('Sharpe ratio', info['sharpe_ratio'], episode)
        self.logger.log_scalar('Maximum drawdown', info['maximum_drawdown'], episode)
        self.logger.log_scalar('Exploration rate', info['exploration_rate'], episode)
        self.logger.log_scalar('Position changes', info['position_changes'], episode)
