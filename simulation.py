import argparse
import pickle

parser = argparse.ArgumentParser(description='Run the deep reinforcement network')
parser.add_argument("--episodes", default=10, help="Number of episodes to simulate")
args = parser.parse_args()

episodes = int(args.episodes)

if __name__ == '__main__':
    from agents import Trader
    from environments import Environment

    with open('data/data.pkl', 'rb') as handle:
        stocks = pickle.load(handle)
        window_size = stocks['^GSPC']['train']['seq'].shape[1]

    # Initialize the deep Q learner
    trader = Trader(state_size=window_size, 
                    discount_rate=.99,
                    sample_batch_size=32)

    # Create the environment for the trader to trade in
    environment = Environment(data=stocks['^GSPC']['train'], 
                              state_size=window_size, 
                              trader=trader,
                              log_file='logs',
                              T=10)

    # Start the simulation
    environment.run(episodes)

    # # Create the environment for the trader to trade in
    # environment = Environment(data=stocks['^GSPC']['test'], 
    #                           state_size=window_size, 
    #                           trader=trader,
    #                           log_file='logs-test',
    #                           T=10)

    # # Start the simulation
    # environment.run(1)