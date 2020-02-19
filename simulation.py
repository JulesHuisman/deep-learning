import argparse
import pickle

parser = argparse.ArgumentParser(description='Run the deep reinforcement network')
parser.add_argument("--window", default=10, help="Number of episodes to simulate")
parser.add_argument("--episodes", default=10, help="Number of episodes to simulate")
parser.add_argument("--train", default=.5, help="Train for what percentage of the data")
args = parser.parse_args()

window_size      = int(args.window)
episodes         = int(args.episodes)
train_percentage = float(args.train)

if __name__ == '__main__':
    from agent import Trader
    from environment import Environment, Stock, rewards

    # Initialize the deep Q learner
    trader = Trader(state_size=window_size,
                    sample_batch_size=64)

    # Get the stock data
    stock = Stock('^GSPC', window_size, train_percentage, True, start_date='2015-1-1', end_date='2020-1-1')

    # Create the environment for the trader to trade in
    environment = Environment(stock=stock, 
                              window_size=window_size, 
                              train_percentage=train_percentage,
                              trader=trader,
                              log_file='logs',
                              reset_trader=True,
                              T=5,
                              reward_type=rewards.LONGTERM)

    # Start the simulation
    environment.run(episodes)

    # # Create the environment for the trader to trade in
    # environment = Environment(data=stocks['^HSI']['test'], 
    #                           state_size=window_size, 
    #                           trader=trader,
    #                           log_file='logs-test',
    #                           T=5)

    # # Start the simulation
    # environment.run(1)