import numpy as np

from utils import normalize

BUY = 0
SELL = 1
HOLD = 2

class Environment:
    def __init__(self,
                 data,
                 state_size,
                 trader,
                 sample_batch_size=64,
                 is_eval=False):

        self.logs = []
        self.profits = 0
        self.data = data
        self.stocks = []
        self.state_size = state_size
        self.sample_batch_size = sample_batch_size
        self.trader = trader
        self.is_eval = is_eval

    def reset(self):
        self.profits = 0
        self.stocks = []

    def buy_stocks(self):
        self.stocks.append(self.stock_price)
        # print(f'Buy: {self.stock_price}')
        return 0

    def sell_stocks(self):
        # If there is no inventory
        if len(self.stocks) == 0:
            return 0

        bought_price = self.stocks.pop(0)
        profit = self.stock_price - bought_price
        self.profits += profit

        # print(f'Sell: {self.stock_price} | Profit: {profit}')

        # Return the profit as reward
        return profit

    def run(self, episodes=1):
        """
        Start the training simulation
        """
        for episode in range(episodes):
            state = self.data[0:1]
            reward = 0
            done = False
            index = 0
            logs = {'profit': [], 'actions': []}

            while not done:
                index += 1

                action = self.trader.get_next_action(normalize(state))
                self.stock_price = state[0][-1]

                if action == BUY:
                    reward = self.buy_stocks()
                elif action == SELL:
                    reward = self.sell_stocks()
                elif action == HOLD:
                    reward = 0

                next_state = self.data[index:index+1]

                self.trader.remember(normalize(state), normalize(next_state), action, reward)
                state = next_state

                logs['profit'].append(self.profits)
                logs['actions'].append(action)

                if (index >= len(self.data) - 1):
                    done = True

            buys = np.sum(np.array(logs['actions']) == BUY)
            sell = np.sum(np.array(logs['actions']) == SELL)
            hold = np.sum(np.array(logs['actions']) == HOLD)

            print(f'Episode: {episode} | Profit: {int(self.profits)} | Buys: {buys} | Sells: {sell} | Holds: {hold} | Exploration: {self.trader.exploration_rate}')

            if not self.is_eval:
                self.trader.replay(self.sample_batch_size)

            self.reset()
            self.logs.append(logs)
