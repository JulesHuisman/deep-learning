import os
import numpy as np

from deepfour import DeepFour
from simulation.game import Game
from simulation.nodes import StateNode
from simulation.mcts import mcts, get_policy
from time import sleep
from random import shuffle
from itertools import cycle
from math import ceil

np.set_printoptions(precision=2, suppress=True, linewidth=150)

class EvaluateProcess:
    """
    https://github.com/Zeta36/connect4-alpha-zero
    """
    def __init__(self, config):
        self.config = config

        # The current best model
        self.best = DeepFour(config)

        # The challenger
        self.challenger = DeepFour(config)

        # Model location
        self.model_location = os.path.join('data', self.config.model, 'models')

        # Challenger location
        self.challenger_location = os.path.join(self.model_location, self.config.model + '.challenger.h5')

    def evaluate(self):
        while True:

            # Check if there is a new challenger
            if not os.path.isfile(self.challenger_location):
                print('No challenger')
                sleep(10)
                continue

            self.best.load('best')
            self.challenger.load('challenger')

            challenger_wins = 0
            win_threshold = ceil(self.config.duel_games * self.config.duel_threshold)

            for game_nr in range(self.config.duel_games):
                game_nr += 1
                challenger_wins += self.duel()
                counter_wins = game_nr - challenger_wins
                print(challenger_wins, '/', game_nr, f'({round(challenger_wins / game_nr, 2)})')
                print(counter_wins, '/', game_nr, f'({round(counter_wins / game_nr, 2)})')

                # The challenger is better
                if challenger_wins >= win_threshold:
                    print('Challenger is better')
                    self.challenger.save('best')
                    break
                # The best (or draw) won
                elif counter_wins >= win_threshold:
                    print('Challenger not good enough')
                    break

            # Delete the challenger
            os.remove(self.challenger_location)

    def duel(self):
        """
        Let two models duel to see which one is better
        """
        # Take turns playing
        nets = [self.best, self.challenger]
        shuffle(nets)
        turns = cycle(nets)
        
        # Create a new empty game
        game       = Game()
        move_count = 0
        
        while True:
            net = next(turns)
            
            # Create a new root node
            root = StateNode(game=game)

            encoded_board = game.encoded()
            policy_pred, value_pred = net.predict(encoded_board)
                
            # Run MCTS
            root = mcts(root, self.config.search_depth, net)
            
            # Get the next policy
            temperature = 0.1
            policy = get_policy(root, temperature)
            
            print('Turn for:', f'\033[94m{net.version}\033[0m', '(\033[95mX\033[0m)' if game.player == -1 else '(\033[92mO\033[0m)', '\n')
            print('Visits:     ', root.child_number_visits, '\n')
            print('Policy:     ', policy)
            print('Policy pred:', policy_pred, '\n')
            print('Value pred:', value_pred, '\n')

            # Decide the next move based on the policy
            move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=policy)
            
            # Make the move
            game.play(move)
            
            game.presentation()
            print()
            
            # If somebody won
            if game.won():
                print('Winner', net.version)
                return net.version == 'challenger'

            if game.moves() == []:
                print('Draw')
                return 0
                
            move_count += 1