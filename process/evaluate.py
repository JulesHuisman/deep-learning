import os
import numpy as np
import mlflow

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

    def evaluate(self, loop=True):
        while True:

            # Check if there is a new challenger
            if not os.path.isfile(self.challenger_location):
                print('No challenger')
                if not loop:
                    return
                sleep(10)
                continue

            try:
                self.best.load('best')
                self.challenger.load('challenger')

                # Delete the challenger
                os.remove(self.challenger_location)

            except:
                continue

            challenger_wins = 0
            challenger_win_threshold = round(self.config.duel_games * self.config.duel_threshold)
            counter_win_threshold = round(self.config.duel_games * (1 - self.config.duel_threshold))

            for game_nr in range(self.config.duel_games):
                game_nr += 1
                challenger_wins += self.duel()
                counter_wins = game_nr - challenger_wins
                print(challenger_wins, '/', challenger_win_threshold, f'({round(challenger_wins / game_nr, 2)})')
                print(counter_wins, '/', counter_win_threshold, f'({round(counter_wins / game_nr, 2)})')
                mlflow.log_metric('win-rate', (challenger_wins / game_nr))

                # The challenger is better
                if challenger_wins >= challenger_win_threshold:
                    print('Challenger is better')
                    try:
                        self.challenger.save('best')
                    except:
                        pass
                    mlflow.log_metric('final-win-rate', (challenger_wins / game_nr))
                    break
                # The best won
                elif counter_wins >= counter_win_threshold:
                    print('Challenger not good enough')
                    mlflow.log_metric('final-win-rate', (challenger_wins / game_nr))
                    break

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
            root = StateNode(game=game, c_puct=1.5, depth=(move_count + 1))

            encoded_board = game.encoded()
            policy_pred, value_pred = net.predict(encoded_board)
                
            # Run MCTS
            root = mcts(root, self.config.search_depth, net)
            
            # Get the next policy
            if move_count <= 1:
                temperature = 2
                policy = get_policy(root, temperature)
                # Decide the next move based on the policy
                move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=policy)
            else:
                move = np.argmax(root.child_number_visits)
                policy = np.zeros(7)
                policy[move] = 1
            
            print('Turn for:', f'\033[94m{net.version}\033[0m', '(\033[95mX\033[0m)' if game.player == -1 else '(\033[92mO\033[0m)', '\n')
            print('Visits:     ', root.child_number_visits, '\n')
            print('Policy:     ', policy)
            print('Policy pred:', root.child_priors, '\n')
            print('Value pred: ', value_pred, '\n')
            
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