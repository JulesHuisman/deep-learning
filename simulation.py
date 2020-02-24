import argparse
import pickle

parser = argparse.ArgumentParser(description='Run the deep reinforcement network')
parser.add_argument("--window", default=200, help="Stock sequence size")
parser.add_argument("--episodes", default=5, help="Number of episodes to simulate")
args = parser.parse_args()

window_size      = int(args.window)
episodes         = int(args.episodes)

if __name__ == '__main__':
    from agent import Trader
    from environment import Environment
    from stock import Stock

    # Initialize the deep Q learner
    trader = Trader(state_size=window_size,
                    neurons=100,
                    neuron_shrink_ratio=.5,
                    hidden_layers=2,
                    discount_rate=.85,
                    learning_rate=0.0001,
                    sample_batch_size=64,
                    target_tau=0.0003)

    # Get the stock data
    stock = Stock(ticker='^GSPC', 
                  window_size=window_size, 
                  train_size=1.0,
                  normalize=True,
                  diff=True,
                  clip=True,
                  start_date='2001-1-1', 
                  end_date='2015-12-31')

    # Create the environment for the trader to trade in
    environment = Environment(stock=stock, 
                              window_size=window_size, 
                              train_percentage=.26,
                              train_iterations=10,
                              trader=trader,
                              log_file='logs',
                              max_drawdown=.3)

    # Start the simulation
    ratio = environment.run()