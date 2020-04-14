import random
import numpy as np
import multiprocessing

from copy import deepcopy
from time import sleep, time
from simulation.nodes import StateNode
from simulation.game import Game
from simulation.mcts import mcts, get_policy
import mlflow

np.set_printoptions(precision=2, suppress=True, linewidth=150)

class SelfPlayProcess:
    def __init__(self, config, memory):
        self.config = config
        self.memory = memory

    def play(self, log):
        from deepfour import DeepFour

        # Perform self play with the best model
        net = DeepFour(self.config, only_predict=True)

        step = 0

        while True:
            random.seed()
            np.random.seed()

            # Store the number of games
            if log:
                mlflow.log_metric('n-games', self.memory.n_games())

                if step % 5 == 0:
                    self.memory.load_memories()
                    mlflow.log_metric('n-moves', len(self.memory.memory))

            net.load('best', log)

            # Create a new empty game
            game = Game()
            
            states        = []  # Storage for all game states
            states_values = []  # Storage for all game states with game outcome
            move_count    = 0
            done          = False
            
            # Keep playing while the game is not done
            while not done:
                # Root of the search tree is the current move
                root = StateNode(game=game, c_puct=self.config.c_puct)

                if log:
                    print('Move:', '\033[95mX\033[0m' if game.player == -1 else '\033[92mO\033[0m', '\n')
                
                # Encoded board
                encoded_board = deepcopy(game.encoded())
                
                # Run UCT search
                root = mcts(root,
                            search_depth=self.config.search_depth,
                            net=net,
                            add_noise=True,
                            noise_eps=self.config.noise_eps,
                            dirichlet_alpha=self.config.dirichlet_alpha)
                
                # Explore for the first n moves, after that exploit
                if move_count <= self.config.exploit_turns:
                    # Get the next policy
                    policy = get_policy(root, 1)
                    # Decide the next move based on the policy
                    move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=policy)
                else:
                    policy = get_policy(root, 0.1)
                    # Decide the next move based on the policy
                    move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=policy)
                    # move = np.argmax(root.child_number_visits)
                    # policy = np.zeros(7)
                    # policy[move] = 1

                # Increase the total moves
                move_count += 1

                # Make the move
                game.play(move)

                # Q value of the board state
                q_value = (root.child_Q()[move])
            
                # Log status to the console
                if log:
                    self.console_print(encoded_board, game, policy, net, q_value, root.child_number_visits)
                
                # Store the intermediate state
                states.append((encoded_board, policy))
                
                # If somebody won
                if game.won():
                    if (game.player * -1) == -1:
                        if log:
                            print('\033[95mX Won!\033[0m \n')
                    if (game.player * -1) == 1:
                        if log:
                            print('\033[92mO Won!\033[0m \n')

                    value = 1

                    # Store board states for training
                    for state in states[:0:-1]:
                        states_values.append((state[0], state[1], value))
                        value *= -1

                    # Empty board has no value
                    states_values.append((states[0][0], states[0][1], 0))
                        
                    # Mark game as done
                    done = True

                # Game was a draw
                elif game.moves() == []:
                    if log:
                        print('Draw! \n')

                    # Set all values to 0
                    for state in states:
                        states_values.append((state[0], state[1], 0))

                    done = True

            # Store games as a side-effect (because of multiprocessing)
            self.memory.remember(states_values[::-1], str(time()).replace('.', '').ljust(17, '0'))

            step += 1

    @staticmethod
    def console_print(encoded_board, game, policy, net, q_value, visits):
        """
        Log status to the console
        """
        # Predict the policy and value of the board state
        policy_estimate, value_estimate = net.predict(encoded_board)

        policy_print      = ['{:.2f}'.format(value) for value in policy]
        policy_pred_print = ['{:.2f}'.format(value) for value in policy_estimate]
        visits_print      = [f'{int(visit)}' for visit in visits]

        print('Q value:    ', round(q_value, 2))
        print('Value pred: ', round(value_estimate, 2), '\n')
        print('Policy pred: |', ' | '.join(policy_pred_print), '|')
        print('Policy:      |', ' | '.join(policy_print), '|\n')
        print('Visits: |', ' | '.join(visits_print), '|\n')

        game.presentation()
        print()