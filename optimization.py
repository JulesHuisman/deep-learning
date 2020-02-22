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
from bayes_opt.util import load_logs

logger = JSONLogger(path="data/bayesian.json")

def simulation(**hyperparams):
    tickers = ['^GSPC', '^HSI', 'AAPL', 'MSFT', 'AMD', 'TSLA']

    for param, value in hyperparams.items():
        print(param, f'{value:.2f}')

    pool = Pool(8)
    ratios = pool.starmap(run, zip(tickers, itertools.repeat(hyperparams), range(len(tickers)))) 
    pool.close() 
    pool.join()

    for ticker, ratio in zip(tickers, ratios):
        print(f'{ticker} - {ratio:.2f}')

    print(np.mean(np.array(ratios)))

    return np.mean(np.array(ratios))

def run(ticker, hyperparams, position):
    neurons             = hyperparams['neurons']
    neuron_shrink_ratio = hyperparams['neuron_shrink_ratio']
    hidden_layers       = hyperparams['hidden_layers']
    learning_rate       = hyperparams['learning_rate']
    sample_batch_size   = hyperparams['sample_batch_size']
    window_size         = hyperparams['window_size']
    price_difference    = hyperparams['price_difference']
    look_back_ratio     = hyperparams['look_back_ratio']
    discount_rate       = hyperparams['discount_rate']
    target_tau          = hyperparams['target_tau']
    target_update_rate  = hyperparams['target_update_rate']

    neurons            = int(round(neurons))
    hidden_layers      = int(round(hidden_layers))
    sample_batch_size  = int(round(sample_batch_size))
    T                  = int(round(T))
    window_size        = int(round(window_size))
    price_look_back    = int(round(look_back_ratio * window_size))
    price_difference   = int(round(price_difference))
    target_update_rate = int(round(target_update_rate))

    # Initialize the deep Q learner
    trader = Trader(state_size=window_size,
                    neurons=neurons,
                    neuron_shrink_ratio=neuron_shrink_ratio,
                    hidden_layers=hidden_layers,
                    learning_rate=learning_rate,
                    discount_rate=discount_rate,
                    sample_batch_size=sample_batch_size,
                    target_tau=target_tau)

    # Get the stock data
    stock = Stock(ticker=ticker, 
                  window_size=window_size, 
                  train_size=.5, 
                  normalize=True, 
                  diff=(price_difference == 1),
                  start_date='2015-1-1', 
                  end_date='2020-1-1',
                  price_look_back=price_look_back)

    # Create the environment for the trader to trade in
    environment = Environment(stock=stock, 
                              window_size=window_size, 
                              train_percentage=.5,
                              trader=trader,
                              log_file=None,
                              reset_trader=True,
                              pbarpos=position,
                              target_update_rate=target_update_rate
                              episodes=5)

    # Start the simulation
    portfolio, baseline, ratio = environment.run()

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
        'learning_rate': (0.0001, 0.001),
        # The size of the memory of the agent (for how long to remember previous trades)
        'sample_batch_size': (8, 128),
        # Size of stock window
        'window_size': (31, 365),
        # Wether to look at absolute or relative prices e.g. (1,10,5) or (1,9,-5)
        'price_difference': (0, 1),
        # At which percentage of the window size to look at the historic price for reward function
        'look_back_ratio': (.1, .9),
        # Discount rate of the Q function
        'discount_rate': (.8, .99),
        # Interpolation ratio of the target network
        'target_tau': (0.005, .2),
        # Interval of target network adjustment
        'target_update_rate': (1, 365)
    }

    optimizer = BayesianOptimization(
        f=simulation,
        pbounds=parameters
    )

    # load_logs(optimizer, logs=['data/bayesian copy.json'])

    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=5,
        n_iter=10,
    )

    print(optimizer.max)

    
