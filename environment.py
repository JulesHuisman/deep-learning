import numpy as np
import pickle
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from utils import split_sequence

np.set_printoptions(suppress=True)

SHORT = -1
HOLD = 0
LONG = 1

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Environment:
    def __init__(self,
                 stock,
                 window_size,
                 trader,
                 train_percentage=.3,
                 normalize_stocks=True,
                 price_look_back=100,
                 log_file='logs',
                 T=10,
                 reset_trader=False):

        self.train_percentage = train_percentage
        self.window_size      = window_size
        self.trader           = trader
        self.price_look_back  = price_look_back
        self.logs             = {'baseline': [], 'portfolio': []}
        self.log_file         = log_file
        self.T                = T
        self.reset_trader     = reset_trader

        self.data_seq         = stock.sequence
        self.data_price       = stock.prices
        self.train_size       = int(self.train_percentage * len(self.data_seq))
        print(self.train_size)

    def reset(self):
        """
        Reset the environment
        """
        self.position  = 0
        self.portfolio = 0
        self.baseline  = 0

        # Clear the memory of the trader
        self.trader.memory.clear()

        # Reset the model of the trader
        if self.reset_trader:
            self.trader.reset_model()

    def reward(self, action):
        """
        Calculate the reward for the Q function
        """
        return (1 + (action-1) * ((self.stock_price - self.stock_price_1) / self.stock_price_1)) * (self.stock_price_1 / self.stock_price_n)

    def act(self, action):
        """
        Calculate new portfolio value, and hold a new position
        """

        # Print the current position the trader is holding
        if self.position == SHORT:
            print('Shorting')
        elif self.position == HOLD:
            print('Holding')
        elif self.position == LONG:
            print('Longing')

        # Calculate new portfolio value
        self.portfolio = self.portfolio + (self.position * (self.stock_price - self.stock_price_1))

        # Hold new position
        self.position = action - 1

    def calculate_stock_prices(self, index):
        """
        Get the stock prices to calculate rewards
        """
        self.stock_price   = self.data_price[(index-1)+(self.window_size)]
        self.stock_price_1 = self.data_price[(index-1)+(self.window_size-1)]
        self.stock_price_n = self.data_price[(index-1)+(self.window_size-self.price_look_back)]

    def store_actions(self, state, next_state):
        # Calculate the different rewards for the different actions
        for action in range(3):
            reward = self.reward(action)
            self.trader.remember(state, action, reward, next_state)

    def is_training(self, index):
        """
        Check if the environment is still in training mode
        """
        return (index < self.train_size)

    def run(self, episodes=1):
        """
        Start the simulation
        """
        for episode in range(episodes):
            seq_len        = len(self.data_seq)
            state          = self.data_seq[0:1]
            done           = False
            portfolio_logs = [0]
            baseline_logs  = [0]
            hold = 0

            # Reset the state before each run
            self.reset()

            for index, _ in enumerate(self.data_seq): # SUBSTRACT ONE
                print(f'--------------------------------- {episode} / {episodes} --- {index} / {seq_len}')

                # Experience replay every T iterations
                if (index % self.T) == 0:
                    pass
                    # self.trader.replay()

                # Get the three flavor of stock prices
                self.calculate_stock_prices(index)

                # Get the next to store in trader memory
                next_state = self.data_seq[index+1:index+2]

                # Store the states and reward in the memory of the trader
                self.store_actions(state, next_state)
                

                if self.is_training(index):
                    print('Training')
                else:
                    # Get the action from the model
                    action = self.trader.get_action(state)

                    # Get the action the trader would take
                    self.act(action)

                    self.baseline += self.stock_price - self.stock_price_1

                    # Store logs
                    portfolio_logs.append(self.portfolio)
                    baseline_logs.append(self.baseline)

                print(f'Portfolio {bcolors.OKBLUE}${"{0:.2f}".format(self.portfolio)}{bcolors.ENDC}')
                print(f'Stock price {bcolors.OKBLUE}${"{0:.2f}".format(self.stock_price)}{bcolors.ENDC}')
                print(f'Delta {bcolors.FAIL if (self.portfolio - self.baseline < 0) else bcolors.OKGREEN}${"{0:.2f}".format(self.portfolio - self.baseline)}{bcolors.ENDC}')

                index += 1
                state = next_state

            # Store logs
            self.logs['portfolio'].append(np.array(portfolio_logs))
            self.logs['baseline'] = baseline_logs
            self.store_logs()

    def store_logs(self):
        with open(f'data/{self.log_file}.pkl', 'wb') as handle:
            pickle.dump(self.logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
class Stock:
    def __init__(self, ticker, window_size, train_size=.3, normalize=True, start_date='2010-1-1', end_date='2020-1-1'):
        self.ticker      = ticker
        self.window_size = window_size
        self.train_size  = train_size
        self.normalize   = normalize
        self.start_date  = start_date
        self.end_date    = end_date
        
        self._fetch_stock()
        
        if self.normalize:
            self._normalize()
            
        self._sequence()
        
    def _fetch_stock(self):
        # Fetch the historic data of the stock
        stock_data = yf.Ticker(self.ticker)
        
        self.prices = stock_data.history(period='1d', 
                                   start=self.start_date, 
                                   end=self.end_date)
        
        self.prices = self.prices.dropna()['Close'].values
        
    def _normalize(self):
        # The size of the train split
        train_split = int(len(self.prices)*self.train_size)
        train_set = self.prices[:train_split]
        
        # Standardscale the data
        scaler = StandardScaler()
        scaler.fit(train_set.reshape(-1,1))
        
        self.prices_norm = scaler.transform(self.prices.reshape(-1,1)).flatten()
        
    def _sequence(self):
        prices = self.prices_norm if hasattr(self, 'prices_norm') else self.prices
        self.sequence = split_sequence(prices, self.window_size)