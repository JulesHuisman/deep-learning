import numpy as np
import pickle

np.set_printoptions(suppress=True)

SELL = 0
HOLD = 1
BUY = 2

class Environment:
    def __init__(self,
                 data,
                 state_size,
                 trader,
                 price_look_back=100,
                 log_file='logs'):

        self.profits         = 0
        self.train_seq       = data['train']
        self.train_price     = data['train_raw']
        self.test_seq        = data['test']
        self.test_price      = data['test_raw']
        self.stocks          = []
        self.state_size      = state_size
        self.trader          = trader
        self.price_look_back = price_look_back
        self.logs            = []
        self.log_file        = log_file

    def reset(self):
        self.profits = 0
        self.stocks = []

        # Clear the memory of the trader
        self.trader.memory.clear()

    def initialize_trader(self):
        """
        Initialize the trader based on the training data
        """
        print(self.train_seq)

    def buy_stock(self, stock_price):
        self.stocks.append(stock_price)

    def sell_stock(self, stock_price):
        # If there is no inventory
        if len(self.stocks) == 0:
            return

        bought_price = self.stocks.pop(0)

        profit = stock_price - bought_price

        self.profits += profit

    def reward(self, action, stock_price, stock_price_1, stock_price_n):
        return (1 + (action-1) * ((stock_price - stock_price_1) / stock_price_1)) * (stock_price_1 / stock_price_n)

    def run(self, episodes=1):
        """
        Start the training simulation
        """
        for episode in range(episodes):
            seq_len = len(self.data_seq)
            state = self.data_seq[0:1]
            done = False
            index = 0
            profits = []

            self.reset()

            while not done:
                next_state = self.data_seq[index+1:index+2]

                print(f'--------------------------------- {episode} / {episodes} --- {index} / {seq_len}')

                stock_price   = self.data_price[(index+1)+(self.state_size-1)]
                stock_price_1 = self.data_price[(index+1)+(self.state_size-2)]
                stock_price_n = self.data_price[(index+1)+(self.state_size-self.price_look_back-1)]

                # Calculate the different rewards for the different actions
                for action in range(3):
                    reward = self.reward(action, stock_price, stock_price_1, stock_price_n)
                    # print('Action', action, 'Reward', reward)
                    self.trader.remember(state, action, reward, next_state)

                action = self.trader.get_action(state)
                if action == SELL:
                    print('Selling', '❌' if len(self.stocks) == 0 else '✔️')
                    self.sell_stock(stock_price)
                elif action == HOLD:
                    print('Holding')
                elif action == BUY:
                    print('Buying')
                    self.buy_stock(stock_price)

                print('Stocks', len(self.stocks))
                print(f'Profit ${int(self.profits)}')

                self.trader.replay()

                # Store the profit
                profits.append(self.profits)

                index += 1
                state = next_state

                if (index >= len(self.data_seq) - 1):
                    done = True

            # Store profit logs
            self.logs.append(np.array(profits))
            self.store_logs()

    def store_logs(self):
        with open(f'data/{self.log_file}.pkl', 'wb') as handle:
            pickle.dump(np.array(self.logs), handle, protocol=pickle.HIGHEST_PROTOCOL)
