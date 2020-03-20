import numpy as np
import os
import random
import multiprocessing
import time

from nodes import DummyNode, Node
from game import Game
from memory import Memory

from copy import deepcopy
from scipy.special import softmax
from itertools import cycle
from random import shuffle

np.set_printoptions(precision=2, suppress=True, linewidth=150)

class Simulation:
    """
    Perform self-play to improve the ConnectNet
    """
    def __init__(self,
                 net_name,
                 games_per_iteration,
                 moves_per_game,
                 memory_size,
                 minibatch_size,
                 training_loops,
                 workers):

        self.net_name            = net_name
        self.games_per_iteration = games_per_iteration
        self.moves_per_game      = moves_per_game
        self.minibatch_size      = minibatch_size
        self.training_loops      = training_loops
        self.workers             = workers

        # Memory stores played games
        self.memory = Memory(folder=f'data/{self.net_name}/memory', size=memory_size)

    @staticmethod
    def add_noise(priors):
        return 0.80 * priors + 0.20 * np.random.dirichlet(np.zeros([len(priors)], dtype=np.float32) + 192)

    @staticmethod
    def mcts(root, moves_per_game, net, add_noise=False):
        """
        Perform one Monte Carlo Tree Search.
        Decides the next play.
        """
        for i in range(moves_per_game):
            # Select the best leaf (exploit or explore)
            leaf = root.select_leaf()

            # # If the leaf is a winning state
            # if leaf.game.won():
            #     print('Root value before', root.child_total_value)
            #     # print('Game Player:', leaf.game.player, 'Leaf player', leaf.player, 'Final leaf state:')
            #     print('Depth at leaf:', leaf.depth, 'Move of leaf:', leaf.move)
            #     leaf.game.presentation()

            #     leaf.backprop(1)
            #     print('Root value after', root.child_total_value)
            #     continue

            # # If the leaf is a draw
            # if leaf.game.moves() == []:
            #     leaf.backprop(0)
            #     continue
            
            # Encode the board for the connect net
            encoded_board = leaf.game.encoded()
            
            # Predict the policy and value of the board state
            policy_estimate, value_estimate = net.predict(encoded_board)
            # policy_estimate, value_estimate = softmax(np.random.uniform(.3, .7, size=(7))), np.tanh(np.random.uniform(-.4, .4))
        
            if leaf.game.won() or leaf.game.moves() == []:
                leaf.backprop(value_estimate)
            else:
                leaf.expand(policy_estimate, add_noise)
                leaf.backprop(value_estimate)
            
        # print('Root value', root.child_total_value)
        return root

    @staticmethod
    def get_policy(root, temperature):
        """
        Policy is based on the number of visits to that node.
        Better nodes get visited more often.
        Normalize the number of moves to transform it into a probablity distribution.
        Temperature is a measure of exploration.
        """
        n_visits = root.child_number_visits ** (1 / temperature)
        return n_visits / sum(n_visits)

    @staticmethod
    def self_play(simulation, game_nr):
        """
        Let the ConnectNet play a number of games against itself.
        Each moves is decided by the Monte Carlo Tree Search.
        Values are seeded by the neural network.
        We need to create the neural network here, because the lack of support for multiprocessing by Tensorflow.
        """
        # Random seed, otherwise all workers have the same random actions
        random.seed()
        np.random.seed()

        from connect_net import ConnectNet

        try:
            # Get the identity of the sub-process
            worker = multiprocessing.current_process()._identity[0]

            # Only one worker should print
            should_print = (worker % simulation.workers == 0)
        # Not multiprocessing
        except:
            should_print = True

        # # Perform self play with the latest neural network
        net = ConnectNet(simulation.net_name, only_predict=True)
        net.load('current', should_print)
        # net = None

        # Create a new empty game
        game = Game()
        
        states        = []  # Storage for all game states
        states_values = []  # Storage for all game states with game outcome
        move_count    = 0
        done          = False
        
        # Keep playing while the game is not done
        while not done:
            # Root of the search tree is the current move
            root = Node(game=game)

            if should_print:
                print(f'Game: {game_nr}', 'Move:', '\033[95mX\033[0m' if game.player == -1 else '\033[92mO\033[0m', '\n')
            
            # Explore for the first 10 moves, after that exploit
            if move_count <= 10:
                temperature = 1.1
            else:
                temperature = 0.1
            
            # Encoded board
            encoded_board = deepcopy(game.encoded())
            
            # Run UCT search
            root = Simulation.mcts(root, simulation.moves_per_game, net, add_noise=True)
            
            # Get the next policy
            policy = Simulation.get_policy(root, temperature)
            
            # Decide the next move based on the policy
            move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=policy)

            # Increase the total moves
            move_count += 1

            # Make the move
            game.play(move)
        
            # Log status to the console
            if should_print:
                q_value = (root.child_Q()[move] * game.player * -1)
                Simulation.console_print(game_nr, game, policy, net, q_value)
            
            # Store the intermediate state
            states.append((encoded_board, policy))
            
            # If somebody won
            if game.won():
                if (game.player * -1) == -1:
                    if should_print:
                        print('\033[95mX Won!\033[0m \n')
                if (game.player * -1) == 1:
                    if should_print:
                        print('\033[92mO Won!\033[0m \n')

                value = 1

                # Store board states for training (alternate )
                for state in states[:0:-1]:
                    states_values.append((state[0], state[1], value))
                    value *= -1

                # Empty board has no value
                states_values.append((states[0][0], states[0][1], 0))
                    
                # Mark game as done
                done = True

            # Game was a draw
            elif game.moves() == []:
                if should_print:
                    print('Draw! \n')

                # Set all values to 0
                for state in states:
                    states_values.append((state[0], state[1], 0))

                done = True

        # Store games as a side-effect (because of multiprocessing)
        # simulation.memory.remember(states_values[::-1], game_nr)

    @staticmethod
    def console_print(game_nr, game, policy, net, q_value):
        """
        Log status to the console
        """
        # Predict the policy and value of the board state
        policy_estimate, value_estimate = net.predict(game.encoded())

        policy_print      = ['{:.2f}'.format(value) for value in policy]
        policy_pred_print = ['{:.2f}'.format(value) for value in policy_estimate]

        print('Q value:    ', round(q_value, 2))
        print('Value pred: ', round(value_estimate, 2), '\n')
        print('Policy:      |', ' | '.join(policy_print), '|')
        print('Policy pred: |', ' | '.join(policy_pred_print), '|\n')

        game.presentation()
        print()

    def replay(self):
        """
        Train the model by replaying from memory
        """
        # Load games from storage into working memory
        self.memory.load_memories()

        # If the memory is not filled yet, continue self play
        if not self.memory.filled:
            print(f'Memory not filled yet ({len(self.memory.memory)})')
            return
        else:
            print('Loaded memory of size:', len(self.memory.memory))

        from connect_net import ConnectNet

        net = ConnectNet(self.net_name)
        net.load('current')

        for training in range(self.training_loops):
            # Sample a minibatch
            boards, policies, values = self.memory.get_minibatch(self.minibatch_size)

            # Train the model
            net.model.fit(boards, [policies, values], batch_size=32, shuffle=False, epochs=1)

        # Save the network (historic)
        net.save(net.latest_iteration + 1)

        # Save the network (current)
        net.save('current')

    # # @staticmethod
    # def duel(simulation):
        """
        Let two models dual to see which one is better.
        """
        print(5)
        return 5
        # from connect_net import ConnectNet

        # current = ConnectNet(simulation.net_name)
        # current.load('current')

        # best = ConnectNet(simulation.net_name)
        # best.load('best')

        # # Take turns playing
        # nets = [current, best]
        # shuffle(nets)
        # turns = cycle(nets)
        
        # # Create a new empty game
        # game       = Game()
        # done       = False
        # move_count = 0
        
        # while not done and game.moves() != []:
        #     net = next(turns)
            
        #     # Create a new root node
        #     root = Node(game=game)
            
        #     # Encoded board
        #     # encoded_board = deepcopy(game.encoded())
                
        #     # Run MCTS
        #     root = Simulation.mcts(root, simulation.moves_per_game, net)
            
        #     # Get the next policy
        #     temperature = 1 if move_count <= 3 else 0.1
        #     policy = Simulation.get_policy(root, temperature)
            
        #     print('Turn for:', net.version)
        #     # print('Visits:    ', root.child_number_visits, '\n')
        #     # print('Policy:    ', policy)
            
        #     # policy_pred, value_pred = net.predict(encoded_board)
            
        #     # print('Net policy:', policy_pred)
        #     # print('Value pred:', value_pred)

        #     # Decide the next move based on the policy
        #     move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=policy)
            
        #     # print('Q value:   ', root.child_Q()[move], '\n')
            
        #     # Make the move
        #     game.play(move)
            
        #     # game.presentation()
        #     # print()
            
        #     # If somebody won
        #     if game.won():
        #         # print('Winner', net.version)
        #         return net.version
                        
        #         # Mark game as done
        #         done = True
                
        #     move_count += 1
            
        # return 'draw'