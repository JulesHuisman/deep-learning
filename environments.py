import numpy as np
import pickle

np.set_printoptions(suppress=True)

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
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Environment:
    def __init__(self,
                 data,
                 state_size,
                 trader,
                 price_look_back=100,
                 log_file='logs',
                 T=10):

        self.data_seq        = data['seq']
        self.data_price      = data['price']
        self.bank_start      = 10000
        self.bank            = self.bank_start
        self.stocks          = 0
        self.state_size      = state_size
        self.trader          = trader
        self.price_look_back = price_look_back
        self.logs            = {'hold': [], 'trading': [], 'stocks': []}
        self.log_file        = log_file
        self.T               = T

    def reset(self):
        """
        Reset the environment
        """
        self.bank = self.bank_start
        self.stocks = 0

        # Clear the memory of the trader
        self.trader.memory.clear()

    def buy_stock(self, stock_price):
        """
        Buy a single stock
        """
        self.bank -= stock_price
        self.stocks += 1

    def sell_stock(self, stock_price):
        """
        Liquify a single stock
        """
        # If there is no inventory
        if self.stocks == 0:
            return

        self.stocks -= 1
        self.bank += stock_price

    def get_portfolio(self, stock_price):
        """
        Get the portfolio value (bank + unrealized stocks)
        """
        return (self.bank + (self.stocks*stock_price))

    def reward(self, action, stock_price, stock_price_1, stock_price_n):
        """
        Calculate the reward for the Q function
        """
        return (1 + (action-1) * ((stock_price - stock_price_1) / stock_price_1)) * (stock_price_1 / stock_price_n)

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
            stock_logs     = []

            # Reset the state before each run
            self.reset()

            while not done:
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

                # Get the action the trader would take
                action = self.trader.get_action(state)

                if action == SELL:
                    print('Selling')
                    self.sell_stock(stock_price)
                elif action == HOLD:
                    print('Holding')
                elif action == BUY:
                    print('Buying')
                    self.buy_stock(stock_price)

                portfolio = self.get_portfolio(stock_price)
                hold = stock_price / self.data_price[self.state_size] * self.bank_start
                
                print('Stocks', self.stocks)
                print(f'Portfolio {bcolors.OKBLUE}${int(portfolio)}{bcolors.ENDC} ({bcolors.FAIL if (portfolio < hold) else bcolors.OKGREEN}${int(portfolio - hold)}{bcolors.ENDC})')

                # Store logs
                portfolio_logs.append(portfolio)
                stock_logs.append(self.stocks)

                index += 1
                state = next_state

                if (index >= len(self.data_seq) - 1):
                    done = True

            # Store logs
            self.logs['trading'].append(np.array(portfolio_logs))
            self.logs['stocks'].append(np.array(stock_logs))
            self.store_logs()

    def store_logs(self):
        with open(f'data/{self.log_file}.pkl', 'wb') as handle:
            self.logs['hold'] = (self.data_price / self.data_price[200]) * self.bank_start
            pickle.dump(self.logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
