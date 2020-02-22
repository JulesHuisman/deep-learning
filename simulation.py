import argparse
import pickle

parser = argparse.ArgumentParser(description='Run the deep reinforcement network')
parser.add_argument("--window", default=200, help="Stock sequence size")
parser.add_argument("--episodes", default=5, help="Number of episodes to simulate")
parser.add_argument("--train", default=.01, help="Train for what percentage of the data")
args = parser.parse_args()

window_size      = int(args.window)
episodes         = int(args.episodes)
train_percentage = float(args.train)

if __name__ == '__main__':
    from agent import Trader
    from environment import Environment, Stock, rewards

    # Initialize the deep Q learner
    trader = Trader(state_size=window_size,
                    neurons=128,
                    neuron_shrink_ratio=.5,
                    hidden_layers=2,
                    discount_rate=.99,
                    learning_rate=0.00025,
                    sample_batch_size=64,
                    target_tau=0.125)

    # Get the stock data
    stock = Stock(ticker='AMD', 
                  window_size=window_size, 
                  train_size=1,
                  price_look_back=100,
                  normalize=True,
                  diff=True,
                  start_date='2000-1-1', 
                  end_date='2019-12-31')

    # Create the environment for the trader to trade in
    environment = Environment(stock=stock, 
                              window_size=window_size, 
                              train_percentage=train_percentage,
                              trader=trader,
                              log_file='logs',
                              reset_trader=True,
                              target_update_rate=10,
                              episodes=episodes)

    # Start the simulation
    portfolio, baseline, ratio = environment.run()

    # # Get the stock data
    # stock = Stock(ticker='^GSPC', 
    #               window_size=window_size, 
    #               train_size=.333,
    #               price_look_back=100,
    #               normalize=True,
    #               diff=True,
    #               start_date='2001-1-1', 
    #               end_date='2015-12-31')

    # # Create the environment for the trader to trade in
    # environment = Environment(stock=stock, 
    #                           window_size=window_size, 
    #                           train_percentage=0.333,
    #                           trader=trader,
    #                           log_file='logs-test',
    #                           reset_trader=False,
    #                           T=50)

    # # Start the simulation
    # portfolio, baseline, ratio = environment.run(1)






    print('Portfolio ratio', portfolio)
    print('Baseline ratio', baseline)
    print('Final ratio', ratio)