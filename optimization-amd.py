from agent import Trader
from environment import Environment
from stock import Stock

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
    neurons             = hyperparams['neurons']
    neuron_shrink_ratio = hyperparams['neuron_shrink_ratio']
    hidden_layers       = hyperparams['hidden_layers']
    learning_rate       = hyperparams['learning_rate']
    sample_batch_size   = hyperparams['sample_batch_size']
    window_size         = hyperparams['window_size']
    price_difference    = hyperparams['price_difference']
    price_clip          = hyperparams['price_clip']
    discount_rate       = hyperparams['discount_rate']
    target_tau          = hyperparams['target_tau']
    train_iterations    = hyperparams['train_iterations']

    neurons            = int(round(neurons))
    hidden_layers      = int(round(hidden_layers))
    sample_batch_size  = int(round(sample_batch_size))
    window_size        = int(round(window_size))
    price_difference   = int(round(price_difference))
    price_clip         = int(round(price_clip))
    target_update_rate = int(round(target_update_rate))
    train_iterations   = int(round(train_iterations))

    # Initialize the deep Q learner
    trader = Trader(state_size=window_size,
                    neurons=neurons,
                    neuron_shrink_ratio=neuron_shrink_ratio,
                    hidden_layers=hidden_layers,
                    discount_rate=discount_rate,
                    learning_rate=learning_rate,
                    sample_batch_size=sample_batch_size,
                    target_tau=target_tau)

    # Get the stock data
    stock = Stock(ticker='AMD', 
                  window_size=window_size, 
                  train_size=1.0,
                  normalize=False,
                  diff=(price_difference == 1),
                  clip=(price_clip == 1),
                  start_date='2000-1-1', 
                  end_date='2017-1-1')

    # Create the environment for the trader to trade in
    environment = Environment(stock=stock, 
                              window_size=window_size, 
                              train_percentage=.5,
                              train_iterations=train_iterations,
                              trader=trader,
                              log_file='logs',
                              max_drawdown=.5)

    # Start the simulation
    ratio = environment.run()

    return ratio

if __name__ == '__main__':
    parameters = {
        # Amount of neurons in the first hidden layer
        'neurons': (8, 256),
        # Amount of hidden layers in the network
        'hidden_layers': (1, 4),
        # Ratio between subsequent hidden layers e.g. (.5 = 128 -> 64 -> 32)
        'neuron_shrink_ratio': (.3, 1.),
        # The learning rate of the neural network
        'learning_rate': (0.0001, 0.001),
        # The size of the memory of the agent (for how long to remember previous trades)
        'sample_batch_size': (8, 128),
        # Size of stock window
        'window_size': (31, 365),
        # Wether to look at absolute or relative prices e.g. (1,10,5) or (1,9,-5)
        'price_difference': (0, 1),
        # Whether to clip the price or not
        'price_clip': (0, 1),
        # Discount rate of the Q function
        'discount_rate': (.8, .99),
        # Interpolation ratio of the target network
        'target_tau': (0.005, .2),
        # How many train iterations
        'train_iterations': (1, 3)
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

    
