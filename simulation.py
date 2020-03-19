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
    def mcts(root, moves_per_game, net):
        """
        Perform one Monte Carlo Tree Search.
        Decides the next play.
        """
        for i in range(moves_per_game):
            # Select the best leaf (exploit or explore)
            leaf = root.select_leaf()

            # If the leaf is a winning state
            if leaf.game.won():
                leaf.backprop(leaf.game.player)
                continue

            # If the leaf is a draw
            if leaf.game.moves() == []:
                leaf.backprop(0)
                continue
            
            # Encode the board for the connect net
            encoded_board = leaf.game.encoded()
            
            # Predict the policy and value of the board state
            policy_estimate, value_estimate = net.predict(encoded_board)
            # policy_estimate, value_estimate = softmax(np.random.uniform(.3, .7, size=(7))), np.tanh(np.random.uniform(-.5, .5))
                
            leaf.expand(policy_estimate)
            leaf.backprop(value_estimate)
            
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
        net.load('current', False)
        # net = None

        # Create a new empty game
        game = Game()
        
        states        = []  # Storage for all game states
        states_values = []  # Storage for all game states with game outcome
        value         = 0
        move_count    = 0
        done          = False
        
        # Keep playing while the game is not done
        while not done and game.moves() != []:
            start = time.time()

            # Root of the search tree is the current move
            root = Node(game=game)
            
            # Explore for the first 10 moves, after that exploit
            if move_count <= 10:
                temperature = 1.1
            else:
                temperature = 0.1
            
            # Encoded board
            encoded_board = deepcopy(game.encoded())
            
            # Run UCT search
            root = Simulation.mcts(root, simulation.moves_per_game, net)
            
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
                time_per_move = (time.time() - start) * 1000
                q_value = (root.child_Q()[move] * game.player * -1)
                Simulation.console_print(game_nr, game, policy, net, time_per_move, q_value)
            
            # Store the intermediate state
            states.append((encoded_board, policy))
            
            # If somebody won
            if game.won():

                if (game.player * -1) == -1:
                    if should_print:
                        print('X won \n')
                    value = -1
                if (game.player * -1) == 1:
                    if should_print:
                        print('O won \n')
                    value = 1
                    
                # Mark game as done
                done = True
        
        # Store the board policy value tuple (used for training)
        for index, data in enumerate(states):
            # Value is zero for an empty board
            if index == 0:
                states_values.append((data[0], data[1], 0))
            else:
                states_values.append((data[0], data[1], value))

        # Store games as a side-effect (because of multiprocessing)
        simulation.memory.remember(states_values, game_nr)

    @staticmethod
    def console_print(game_nr, game, policy, net, time_per_move, q_value):
        """
        Log status to the console
        """
        # Predict the policy and value of the board state
        policy_estimate, value_estimate = net.predict(game.encoded())

        policy_print      = ['{:.2f}'.format(value) for value in policy]
        policy_pred_print = ['{:.2f}'.format(value) for value in policy_estimate]

        print(f'Game: {game_nr}', 'Move:', '\033[95mX\033[0m' if game.player == 1 else '\033[92mO\033[0m', f'({round(time_per_move)} ms)', '\n')

        # print('Value:  ', round(value_estimate, 2),)
        print('Q value:', round(q_value, 2), '\n')
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