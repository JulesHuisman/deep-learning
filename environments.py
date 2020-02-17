import numpy as np
import pickle

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
                 data,
                 state_size,
                 trader,
                 price_look_back=100,
                 log_file='logs',
                 T=10,
                 stock_multiplier=1,
                 reset_trader=False):

        self.data_seq         = data['seq']
        self.data_price       = data['price']
        self.position         = 0
        self.portfolio        = 0
        self.stock_multiplier = stock_multiplier
        self.state_size       = state_size
        self.trader           = trader
        self.price_look_back  = price_look_back
        self.logs             = {'hold': [], 'portfolio': []}
        self.log_file         = log_file
        self.T                = T
        self.reset_trader     = reset_trader

    def reset(self):
        """
        Reset the environment
        """
        self.position = 0
        self.portfolio = 0

        # Clear the memory of the trader
        self.trader.memory.clear()

        # Reset the model of the trader
        if self.reset_trader:
            self.trader.reset_model()

    def reward(self, action, stock_price, stock_price_1, stock_price_n):
        """
        Calculate the reward for the Q function
        """
        return (1 + (action-1) * ((stock_price - stock_price_1) / stock_price_1)) * (stock_price_1 / stock_price_n)

    def act(self, action, stock_price, stock_price_1):
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

        market_delta = (stock_price - stock_price_1)

        # Calculate new portfolio value
        self.portfolio = self.portfolio + (self.position * self.stock_multiplier * market_delta)

        # Hold new position
        self.position = action - 1

    def run(self, episodes=1):
        """
        Start the simulation
        """
        for episode in range(episodes):
            seq_len        = len(self.data_seq)
            state          = self.data_seq[0:1]
            done           = False
            index          = 0
            portfolio_logs = []

            # Reset the state before each run
            self.reset()

            for _ in self.data_seq:
                # Experience replay every T iterations
                if index % self.T == 0:
                    self.trader.replay()

                # Log iteration
                print(f'--------------------------------- {episode} / {episodes} --- {index} / {seq_len}')

                # Get the stock prices to calculate rewards
                stock_price   = self.data_price[index+(self.state_size)]
                stock_price_1 = self.data_price[index+(self.state_size-1)]
                stock_price_n = self.data_price[index+(self.state_size-self.price_look_back)]

                # Get the next to store in trader memory
                next_state = self.data_seq[index+1:index+2]

                # Calculate the different rewards for the different actions
                for action in range(3):
                    reward = self.reward(action, stock_price, stock_price_1, stock_price_n)
                    self.trader.remember(state, action, reward, next_state)

                # Get the action the trader would take (only act when the trader is learning)
                if self.trader.memory_filled():
                    action = self.trader.get_action(state)
                    self.act(action, stock_price, stock_price_1)

                hold = stock_price - self.data_price[self.state_size]

                print(f'Portfolio {bcolors.FAIL if (self.portfolio < 0) else bcolors.OKGREEN}${"{0:.2f}".format(self.portfolio)}{bcolors.ENDC}')
                print(f'Delta {bcolors.OKBLUE}${"{0:.2f}".format(self.portfolio - hold)}{bcolors.ENDC}')

                # Store logs
                portfolio_logs.append(self.portfolio)

                index += 1
                state = next_state

            # Store logs
            self.logs['portfolio'].append(np.array(portfolio_logs))
            self.store_logs()

    def store_logs(self):
        with open(f'data/{self.log_file}.pkl', 'wb') as handle:
            self.logs['hold'] = self.data_price - self.data_price[200]
            pickle.dump(self.logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
