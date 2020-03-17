import numpy as np
import os
import random
import multiprocessing

from nodes import DummyNode, Node
from game import Game
from memory import Memory

from itertools import repeat
from copy import deepcopy
from multiprocessing import Pool

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
                 workers,
                 games_per_pool):

        self.net_name            = net_name
        self.games_per_iteration = games_per_iteration
        self.moves_per_game      = moves_per_game
        self.minibatch_size      = minibatch_size
        self.training_loops      = training_loops
        self.workers             = workers
        self.games_per_pool      = games_per_pool

        # Memory stores played games
        self.memory = Memory(folder=f'data/{self.net_name}/memory', size=memory_size)

    @staticmethod
    def mcts(root, moves_per_game, net):
        """
        Perform one Monte Carlo Tree Search.
        Decides the next play.
        """
        # Add some noise to the children of the root node
        root.add_dirichlet_noise()

        for i in range(moves_per_game):
            # Select the best leaf (exploit or explore)
            leaf = root.select_leaf()
            
            # Encode the board for the connect net
            encoded_board = leaf.game.encoded()
            
            # Predict the policy and value of the board state
            policy_estimate, value_estimate = net.predict(np.array([encoded_board]))
            policy_estimate, value_estimate = policy_estimate[0], value_estimate[0][0]
            
            # If end is reached (somebody won or draw)
            if leaf.game.won() or leaf.game.moves() == []:
                leaf.backup(value_estimate)
                continue
                
            leaf.expand(policy_estimate)
            leaf.backup(value_estimate)
            
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
    def self_play(simulation, start_game_nr, game_nr):
        """
        Let the ConnectNet play a number of games against itself.
        Each moves is decided by the Monte Carlo Tree Search.
        Values are seeded by the neural network.
        We need to create the neural network here, because the lack of support for multiprocessing by Tensorflow.
        """
        from connect_net import ConnectNet

        # Get the real game number
        game_nr = start_game_nr + game_nr

        # Get the identity of the sub-process
        worker = multiprocessing.current_process()._identity[0]
        print(f'Worker {worker} is picking up game {game_nr}')

        random.seed()
        np.random.seed()

        # Perform self play with the latest neural network
        net = ConnectNet(simulation.net_name)
        net.load('current')

        # Create a new empty game
        game = Game()
        
        states        = []  # Storage for all game states
        states_values = []  # Storage for all game states with game outcome
        value         = 0
        move_count    = 0
        done          = False

        root = Node(game=game,
                    move=None,
                    parent=DummyNode())
        
        # Keep playing while the game is not done
        while not done and root.game.moves() != []:
            
            # Explore less when later in the game
            if move_count <= 10:
                temperature = 1.1
            else:
                temperature = 0.1
            
            # Encoded board
            encoded_board = deepcopy(root.game.encoded())
            
            # Run UCT search
            root = Simulation.mcts(root, simulation.moves_per_game, net.model)
            
            # Get the next policy
            policy = Simulation.get_policy(root, temperature)
            
            # Decide the next move based on the policy
            move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=policy)

            # Update the root node
            root = root.children[move]
        
            # Log status to the console
            if worker == 1:
                # pass
                Simulation.console_print(game_nr, root, policy, net)
            
            # Store the intermediate state
            states.append((encoded_board, policy))
            
            # If somebody won
            if root.game.won():

                if (root.game.player * -1) == -1:
                    if worker == 1:
                        print('X won \n')
                    value = -1
                if (root.game.player * -1) == 1:
                    if worker == 1:
                        print('O won \n')
                    value = 1
                    
                # Mark game as done
                done = True
            
            # Increase the total moves
            move_count += 1
        
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
    def console_print(game_nr, root, policy, net):
        """
        Log status to the console
        """
        # Predict the policy and value of the board state
        policy_estimate, value_estimate = net.model.predict(np.array([root.game.encoded()]))

        policy_print      = ['{:.2f}'.format(value) for value in policy]
        policy_pred_print = ['{:.2f}'.format(value) for value in policy_estimate[0]]

        print(f'Game: {game_nr}', 'Move:', '\033[95mX\033[0m (-1)' if root.game.player == 1 else '\033[92mO\033[0m (1)', '\n')

        print('Policy:     ', ' | '.join(policy_print))
        print('Policy pred:', ' | '.join(policy_pred_print), '\n')

        print('Q Value:   ', round(root.total_value / root.parent.child_number_visits[root.move], 2))
        print('Value pred:', round(value_estimate[0][0], 2), '\n')

        root.game.presentation()
        print()

    def replay(self, iteration):
        """
        Train the model by replaying from memory
        """
        from connect_net import ConnectNet
        
        print('Before', len(self.memory.memory))

        # Load games from storage into working memory
        self.memory.load_memories()

        print('After', len(self.memory.memory))

        # If the memory is not filled yet, continue self play
        if not self.memory.filled:
            print(f'Memory not filled yet ({len(self.memory.memory)})')
            return

        net = ConnectNet(self.net_name)

        for training in range(self.training_loops):
            # Sample a minibatch
            boards, policies, values = self.memory.get_minibatch(self.minibatch_size)

            print(values)

            # Train the model
            net.model.fit(boards, [policies, values], batch_size=self.minibatch_size, shuffle=False, epochs=1)

        # Save the network (historic)
        net.save(iteration)

        # Save the network (current)
        net.save('current')

    def run(self):
        """
        Run the simulation
        """
        # Start at the latest game
        game_nr = self.memory.latest_game + 1

        # Keep track of the iteration for training
        prev_iteration = game_nr // self.games_per_iteration

        print('Iteration', prev_iteration)

        # Start a pool of workers
        with Pool(processes=self.workers) as pool:
            while True:
                # Distribute multiple self play instances over the pool of workers
                pool.starmap(Simulation.self_play, zip(repeat(self), repeat(game_nr), range(self.games_per_pool)))
                
                # Increase the current game number
                game_nr = self.memory.latest_game + 1

                iteration = game_nr // self.games_per_iteration

                print('Iteration', iteration)

                # When arriving at a new iteration, retrain the network
                if iteration > prev_iteration:
                    # Train the model on self play games
                    self.replay(iteration=iteration)

                    # Update previous iteration
                    prev_iteration = iteration

                    # Dual against best