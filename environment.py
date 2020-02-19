import numpy as np
import pickle
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from utils import split_sequence

np.set_printoptions(suppress=True)


class actions:
    SHORT = -1
    HOLD = 0
    LONG = 1

class rewards:
    SHORTTERM = 0
    LONGTERM = 1

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
                 normalize_stocks=True,
                 log_file='logs',
                 T=10,
                 reset_trader=False):

        self.stock            = stock
        self.train_percentage = train_percentage
        self.window_size      = window_size
        self.trader           = trader
        self.logs             = {'baseline': [], 'portfolio': []}
        self.log_file         = log_file
        self.T                = T
        self.reset_trader     = reset_trader

        self.train_size       = int(self.train_percentage * len(self.stock.stock_prices))
        self.start_stock      = self.stock.stock_prices[self.train_size]

    def reset(self):
        """
        Reset the environment
        """
        self.position  = 0
        self.portfolio = self.start_stock
        self.baseline  = self.start_stock

        # Clear the memory of the trader
        self.trader.memory.clear()

        # Reset the model of the trader
        if self.reset_trader:
            self.trader.reset_model()

    def reward(self, action):
        """
        Calculate the reward for the Q function
        """
        
        current_stock_price = self.stock.stock_prices[self.index]
        prev_stock_price    = self.stock.stock_prices_1[self.index]
        prev_n_stock_price  = self.stock.stock_prices_n[self.index]

        return (1 + (action-1) * ((current_stock_price - prev_stock_price) / prev_stock_price)) * (prev_stock_price / prev_n_stock_price)

    def act(self, action):
        """
        Calculate new portfolio value, and hold a new position
        """

        # Print the previous position the trader is holding
        if self.position == actions.SHORT:
            print('Shorting')
        elif self.position == actions.HOLD:
            print('Holding')
        elif self.position == actions.LONG:
            print('Longing')

        current_stock_price = self.stock.stock_prices[self.index]
        prev_stock_price    = self.stock.stock_prices_1[self.index]

        # Calculate new portfolio value
        self.portfolio = self.portfolio + (self.position * (current_stock_price - prev_stock_price))

        # Hold new position
        self.position = action - 1

    def store_actions(self, state, next_state):
        # Calculate the different rewards for the different actions
        for action in range(3):
            reward = self.reward(action)
            self.trader.remember(state, action, reward, next_state)

    def is_training(self):
        """
        Check if the environment is still in training mode
        """
        return (self.index < self.train_size)

    def run(self, episodes=1):
        """
        Start the simulation
        """
        for episode in range(episodes):
            seq_len        = len(self.stock.stock_seq)
            state          = self.stock.stock_seq[0:1]
            done           = False
            portfolio_logs = []
            baseline_logs  = []
            hold = 0

            # Reset the state before each run
            self.reset()

            for index in range(seq_len):
                # Set the index of the iteration
                self.index = index

                # Stock prices
                current_stock_price = self.stock.stock_prices[self.index]
                prev_stock_price    = self.stock.stock_prices_1[self.index]

                print(f'--------------------- {episode} / {episodes} --- {self.index} / {seq_len}')

                # Experience replay every T iterations
                if (index % self.T) == 0:
                    self.trader.replay()

                # Get the next state to store in trader memory
                next_state = self.stock.stock_seq[index+1:index+2]

                # Store the states and reward in the memory of the trader
                self.store_actions(state, next_state)
                
                if self.is_training():
                    print('Training')
                else:
                    # Get the action from the model
                    action = self.trader.get_action(state)

                    # Get the action the trader would take
                    self.act(action)

                    # Baseline is longing the stock
                    self.baseline += current_stock_price - prev_stock_price

                    # Store logs
                    portfolio_logs.append(self.portfolio / self.start_stock)
                    baseline_logs.append(self.baseline / self.start_stock)

                print(f'Portfolio {bcolors.OKBLUE}${"{0:.2f}".format(self.portfolio)}{bcolors.ENDC}')
                print(f'Stock price {bcolors.OKBLUE}${"{0:.2f}".format(current_stock_price)}{bcolors.ENDC}')
                print(f'Delta {bcolors.FAIL if (self.portfolio - self.baseline < 0) else bcolors.OKGREEN}${"{0:.2f}".format(self.portfolio - self.baseline)}{bcolors.ENDC}')

                state = next_state

            # Store logs
            self.logs['portfolio'].append(np.array(portfolio_logs))
            self.logs['baseline'] = baseline_logs
            self.store_logs()

        return np.average(np.array([p[-1] for p in self.logs['portfolio']]))

    def store_logs(self):
        with open(f'data/{self.log_file}.pkl', 'wb') as handle:
            pickle.dump(self.logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Stock:
    def __init__(self, 
                 ticker, 
                 window_size, 
                 train_size=.3, 
                 normalize=True,
                 diff=True,
                 price_look_back=2,
                 start_date='2010-1-1', 
                 end_date='2020-1-1'):
        
        self.ticker          = ticker
        self.window_size     = window_size
        self.train_size      = train_size
        self.start_date      = start_date
        self.end_date        = end_date
        self.price_look_back = price_look_back
        self.diff            = diff
        self.normalize       = normalize
        
        self.stock_prices_raw = np.array([])
        self.stock_prices     = np.array([])
        self.stock_prices_1   = np.array([])
        self.stock_prices_n   = np.array([])
        self.stock_seq        = np.array([])
        
        self._fetch_stock()
        self._set_stocks()
        self._sequence()
        
    def _fetch_stock(self):
        # Fetch the historic data of the stock
        stock_data = yf.Ticker(self.ticker)
        
        self.stock_prices_raw = stock_data.history(period='1d', 
                                   start=self.start_date, 
                                   end=self.end_date)
        
        self.stock_prices_raw = self.stock_prices_raw.dropna()['Close'].values
        
    def _set_stocks(self):
        """
        Create different stock array that follow the same index as the sequence data
        """
        self.stock_prices = np.array(self.stock_prices_raw[(self.window_size-1):])
        self.stock_prices_1 = np.array(self.stock_prices_raw[(self.window_size-2):])
        self.stock_prices_n = np.array(self.stock_prices_raw[(self.window_size-1-self.price_look_back):])
        
    def _diff(self, stocks):
        return np.diff(stocks, 1, prepend=[stocks[0]])
        
    def _normalize(self, stocks):
        # The size of the train split
        train_split = int(len(stocks)*self.train_size)
        train_set = stocks[:train_split]
        
        # Standardscale the data
        scaler = StandardScaler()
        scaler.fit(train_set.reshape(-1,1))
        
        return scaler.transform(stocks.reshape(-1,1)).flatten()
        
    def _sequence(self):
        stocks = self.stock_prices_raw
        
        if self.diff:
            stocks = self._diff(stocks)
        
        if self.normalize:
            stocks = self._normalize(stocks)
        
        self.stock_seq = split_sequence(stocks, self.window_size)