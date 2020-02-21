from agent import Trader
from environment import Environment, Stock, rewards

import argparse
import pickle
import numpy as np
import itertools

from multiprocessing import Pool, Process

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

logger = JSONLogger(path="data/bayesian.json")

def simulation(**hyperparams):
    tickers = ['^GSPC', '^HSI', 'AAPL', 'MSFT', 'AMD', 'TSLA']

    pool = Pool(6)
    ratios = pool.starmap(run, zip(tickers, itertools.repeat(hyperparams), range(len(tickers)))) 
    pool.close() 
    pool.join()

    for ticker, ratio in zip(tickers, ratios):
        print(f'{ticker} - {ratio:.2f}')

    return np.mean(np.array(ratios))

def run(ticker, hyperparams, position):
    neurons             = hyperparams['neurons']
    neuron_shrink_ratio = hyperparams['neuron_shrink_ratio']
    hidden_layers       = hyperparams['hidden_layers']
    learning_rate       = hyperparams['learning_rate']
    sample_batch_size   = hyperparams['sample_batch_size']
    train_size          = hyperparams['train_size']
    T                   = hyperparams['T']
    window_size         = hyperparams['window_size']
    price_difference    = hyperparams['price_difference']
    look_back_ratio     = hyperparams['look_back_ratio']
    discount_rate       = hyperparams['discount_rate']

    neurons           = int(neurons)
    hidden_layers     = int(hidden_layers)
    sample_batch_size = int(sample_batch_size)
    T                 = int(T)
    window_size       = int(window_size)
    price_look_back   = int(look_back_ratio * window_size)
    price_difference  = round(price_difference)

    # Initialize the deep Q learner
    trader = Trader(state_size=window_size,
                    neurons=neurons,
                    neuron_shrink_ratio=neuron_shrink_ratio,
                    hidden_layers=hidden_layers,
                    learning_rate=learning_rate,
                    discount_rate=discount_rate,
                    sample_batch_size=sample_batch_size)

    # Get the stock data
    stock = Stock(ticker=ticker, 
                  window_size=window_size, 
                  train_size=train_size, 
                  normalize=True, 
                  diff=(price_difference == 1),
                  start_date='2012-1-1', 
                  end_date='2020-1-1',
                  price_look_back=price_look_back)

    # Create the environment for the trader to trade in
    environment = Environment(stock=stock, 
                              window_size=window_size, 
                              train_percentage=train_size,
                              trader=trader,
                              log_file=None,
                              reset_trader=True,
                              T=T,
                              pbarpos=position)

    # Start the simulation
    portfolio, baseline, ratio = environment.run(3)

    return ratio

if __name__ == '__main__':
    parameters = {
        # Amount of neurons in the first hidden layer
        'neurons': (8, 128),
        # Amount of hidden layers in the network
        'hidden_layers': (1, 3),
        # Ratio between subsequent hidden layers e.g. (128 * .5 = 64)
        'neuron_shrink_ratio': (.3, 1.),
        # The learning rate of the neural network
        'learning_rate': (0.0001, 0.01),
        # The size of the memory of the agent (for how long to remember previous trades)
        'sample_batch_size': (8, 64),
        # For how long to train before trading starts
        'train_size': (.1,.5),
        # Memory replay interval
        'T': (1,10),
        # Size of stock window
        'window_size': (200, 200),
        # Wether to look at absolute or relative prices e.g. (1,10,5) or (1,9,-5)
        'price_difference': (0, 1),
        # At which percentage of the window size to look at the historic price for reward function
        'look_back_ratio': (.5, .5),
        # Discount rate of the Q function
        'discount_rate': (.5, .99)
    }

    optimizer = BayesianOptimization(
        f=simulation,
        pbounds=parameters
    )

    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=3,
    )

    print(optimizer.max)

    