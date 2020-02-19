from multiprocessing import Pool, Process
import argparse
import pickle
from agent import Trader
from environment import Environment, Stock, rewards

def run(ticker):
    train_percentage = .5
    window_size      = 200

    # Initialize the deep Q learner
    trader = Trader(state_size=window_size,
                    sample_batch_size=64)

    # Get the stock data
    stock = Stock(ticker=ticker, 
                  window_size=window_size, 
                  train_size=train_percentage, 
                  normalize=True, 
                  diff=True,
                  start_date='2010-1-1', 
                  end_date='2020-1-1')

    # Create the environment for the trader to trade in
    environment = Environment(stock=stock, 
                              window_size=window_size, 
                              train_percentage=train_percentage,
                              trader=trader,
                              log_file=f'logs-{ticker}',
                              reset_trader=True,
                              T=5)

    # Start the simulation
    score = environment.run(3)

    return (ticker, score)

if __name__ == '__main__':
    with Pool(16) as p:
        print(p.map(run, ['^GSPC', 'AAPL', 'MSFT', 'AMD']))