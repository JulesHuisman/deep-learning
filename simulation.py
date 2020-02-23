import argparse
import pickle

parser = argparse.ArgumentParser(description='Run the deep reinforcement network')
parser.add_argument("--window", default=365, help="Stock sequence size")
parser.add_argument("--episodes", default=5, help="Number of episodes to simulate")
parser.add_argument("--train", default=.1, help="Train for what percentage of the data")
args = parser.parse_args()

window_size      = int(args.window)
episodes         = int(args.episodes)
train_percentage = float(args.train)

if __name__ == '__main__':
    from agent import Trader
    from environment import Environment
    from stock import Stock

    # Initialize the deep Q learner
    trader = Trader(state_size=window_size,
                    neurons=128,
                    neuron_shrink_ratio=.8,
                    hidden_layers=4,
                    discount_rate=.99,
                    learning_rate=0.00025,
                    sample_batch_size=64,
                    target_tau=0.05)

    # Get the stock data
    stock = Stock(ticker='^GSPC', 
                  window_size=window_size, 
                  train_size=1.0,
                  normalize=False,
                  diff=True,
                  clip=False,
                  start_date='2000-1-1', 
                  end_date='2017-1-1')

    # Create the environment for the trader to trade in
    environment = Environment(stock=stock, 
                              window_size=window_size, 
                              train_percentage=.5,
                              train_iterations=3,
                              trader=trader,
                              log_file='logs',
                              max_drawdown=.3)

    # Start the simulation
    ratio = environment.run()