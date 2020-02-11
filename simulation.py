import argparse

parser = argparse.ArgumentParser(description='Run the deep reinforcement network')
parser.add_argument("--window", default=14, help="Window size of the stocks")
parser.add_argument("--episodes", default=20, help="Number of episodes to simulate")
args = parser.parse_args()

window_size = int(args.window)
episodes    = int(args.episodes)

if __name__ == '__main__':
    from agents import DQN
    from environments import Environment
    from metaflow import Flow, get_metadata

    run = Flow('TraderFlow').latest_successful_run
    stocks = run.data.stocks

    # Initialize the deep Q learner
    trader = DQN(state_size=window_size)

    # Create the environment for the trader to trade in
    environment = Environment(data=stocks['MSFT'], 
                                state_size=window_size, 
                                trader=trader)

    # Start the simulation
    environment.run(episodes)

    # Evaluation
    print('\n Evaluation --------------------------------------------------- \n')

    # Set to evaluation mode
    trader.is_eval = True
    
    # Create evaluation environment
    val_environment = Environment(data=stocks['A'], 
                                state_size=window_size, 
                                trader=trader,
                                is_eval=True)

    val_environment.run()