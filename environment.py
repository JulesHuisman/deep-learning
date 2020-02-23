import numpy as np
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm, trange

from logger import Logger

np.set_printoptions(suppress=True)

SHORT = 0
NONE = 1
LONG = 2

SELL = 0
HOLD = 1
BUY = 2

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

class Environment:
    def __init__(self,
                 stock,
                 window_size,
                 trader,
                 train_percentage=.3,
                 train_iterations=5,
                 logging=True,
                 log_file='logs',
                 target_update_rate=100,
                 max_drawdown=.7):

        self.stock              = stock
        self.train_percentage   = train_percentage
        self.train_iterations   = train_iterations
        self.window_size        = window_size
        self.trader             = trader
        self.logs               = {'train': [], 'test': []}
        self.logging            = logging
        self.log_file           = log_file
        self.max_drawdown       = max_drawdown
        self.runs               = 0

        # if self.logging:
        #     self.logger = Logger(self.stock.ticker, self.episodes)

        self.train_size        = int(self.train_percentage * len(self.stock.stock_prices))
        self.start_stock_test  = self.stock.stock_prices[self.train_size]
        self.start_stock_train = self.stock.stock_prices[0]

    def reset(self, price):
        """
        Reset the environment.
        """
        self.position    = 1
        self.portfolio   = price
        self.baseline    = price

        # Clear the memory of the trader
        self.trader.memory.clear()

    def reward(self, action):
        """
        Calculate the reward for the Q function.
        """

        current_stock_price = self.stock.stock_prices[self.index]
        prev_stock_price    = self.stock.stock_prices_1[self.index]

        new_portfolio = self.portfolio + ((action-1) * (current_stock_price - prev_stock_price))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.log(new_portfolio / self.portfolio)
        

    def _trade(self, action):
        """
        Calculate new portfolio value, and hold a new position.
        Important to hold a new position after the trade has been done.
        Otherwise you can trade on future information.
        """

        current_stock_price = self.stock.stock_prices[self.index]
        prev_stock_price    = self.stock.stock_prices_1[self.index]

        # print('Position', self.position)
        # print('Prev stock', prev_stock_price)
        # print('Current stock', current_stock_price)
        # print('Price difference', current_stock_price - prev_stock_price)
        # print('Current portfolio', self.portfolio)
        # print('Next portfolio', self.portfolio + (self.position * (current_stock_price - prev_stock_price)))

        # Calculate new portfolio value
        self.portfolio = self.portfolio + (self.position * (current_stock_price - prev_stock_price))

        # Hold new position
        self.position = action - 1

    def _store_in_memory(self, state, next_state):
        """
        Calculate the different rewards for the different actions.
        """

        for action in range(3):
            reward = self.reward(action)
            # print(state[0][-5:], action, reward, next_state[0][-5:])
            self.trader.remember(state, action, reward, next_state)

    def is_training(self):
        """
        Check if the environment is still in training mode.
        """
        return (self.index < self.train_size)

    def _log(self, episode):
        """
        Log the progress of the run.
        """
        if self.logging:
            pass
            # self.logger.log_scalar(episode, 'portfolio ratio', (self.portfolio / self.baseline), self.index)

    def episode(self, episode, start_index, stop_index, start_price, ratio_threshold=None, batch_size=16):
        """
        Run one episode
        """
        self.runs += 1
        self.reset(start_price)

        logs = []

        with tqdm(range(start_index, stop_index)) as progress:
            for index in progress:
                self.index = index

                # Progressbar
                progress.set_postfix(portfolio=f'{self.portfolio / start_price:.3f}',
                                # position=f'{self.position}',
                                ratio=f'{bcolors.FAIL if (self.portfolio / self.baseline < 1) else bcolors.OKGREEN}{(self.portfolio / self.baseline):.3f}{bcolors.ENDC}',
                                episode=(episode+1), 
                                stock=self.stock.ticker)

                # Variable needed for this iteration
                state            = self.stock.stock_seq[index:index+1]
                next_state       = self.stock.stock_seq[index+1:index+2]
                stock_price      = self.stock.stock_prices[index]
                prev_stock_price = self.stock.stock_prices_1[index]

                # Soft update the target network
                self.trader.soft_update_target_model()

                # Get the action that the trader would take
                action = self.trader.get_action(state)

                # Perform a trade based on the action
                self._trade(action)

                # Baseline is longing the stock
                self.baseline += (stock_price - prev_stock_price)

                # Create augmented states and store them in the traders memory
                self._store_in_memory(state, next_state)

                # Replay on the augmented trades
                self.trader.replay()

                # New state is the next state
                state = next_state

                # Stop the episode if minimum threshold is reached
                if ratio_threshold and ((self.portfolio / self.baseline) < ratio_threshold) and self.trader.memory_filled():
                    self.runs -= 1
                    break

                logs.append(self.portfolio / start_price)

            return np.array(logs)

    def run(self):
        """
        Start the simulation
        """
        # for episode in range(self.train_iterations):
        while self.runs < self.train_iterations:
            logs = self.episode(episode=self.runs,
                         start_index=0, 
                         stop_index=self.train_size, 
                         start_price=self.start_stock_train,
                         ratio_threshold=self.max_drawdown)

            # Store logs
            self.logs['train'].append(logs)
            self._store_logs()

        logs = self.episode(episode=0,
                     start_index=self.train_size, 
                     stop_index=len(self.stock.stock_prices)-1,
                     start_price=self.start_stock_test,
                     batch_size=16)

        # Store logs
        self.logs['test'].append(logs)
        self._store_logs()

        return (self.portfolio / self.baseline)

    def _store_logs(self):
        if self.logging:

            self.logs['train_baseline'] = self.stock.stock_prices[:self.train_size] / self.start_stock_train
            self.logs['test_baseline'] = self.stock.stock_prices[self.train_size:] / self.start_stock_test

            with open(f'data/{self.log_file}.pkl', 'wb') as handle:
                pickle.dump(self.logs, handle, protocol=pickle.HIGHEST_PROTOCOL)