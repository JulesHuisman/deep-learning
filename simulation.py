import numpy as np
import os

from nodes import DummyNode, Node
from game import Game
from memory import Memory

from copy import deepcopy
from tempfile import TemporaryFile
from multiprocessing import Pool

np.set_printoptions(precision=2, suppress=True, linewidth=150)

class Simulation:
    """
    Perform self-play to improve the ConnectNet
    """
    def __init__(self, net, games_per_iteration, moves_per_game, memory_size, minibatch_size, training_loops):
        self.games_per_iteration = games_per_iteration
        self.net                 = net
        self.moves_per_game      = moves_per_game
        self.minibatch_size      = minibatch_size
        self.training_loops      = training_loops

        # Stores played games
        self.memory = Memory(folder=f'data/{self.net.name}/memory',
                             size=memory_size)

    def mcts(self, root, moves_per_game, net):
        """
        Perform one Monte Carlo Tree Search.
        Decides the next play.
        """
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

    def get_policy(self, root, temperature):
        """
        Policy is based on the number of visits to that node.
        Better nodes get visited more often.
        Normalize the number of moves to transform it into a probablity distribution.
        Temperature is a measure of exploration.
        """
        n_visits = root.child_number_visits ** (1 / temperature)
        return n_visits / sum(n_visits)

    def self_play(self, game_nr):
        """
        Let the ConnectNet play a number of games against itself.
        Each moves is decided by the Monte Carlo Tree Search.
        Values are seeded by the neural network.
        """
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
        while not done and game.moves() != []:
            
            # Explore less when later in the game
            if move_count <= 10:
                temperature = 1.1
            else:
                temperature = 0.1
            
            # Encoded board
            encoded_board = deepcopy(game.encoded())
            
            # Run UCT search
            root = self.mcts(root, self.moves_per_game, self.net.model)
            
            # Get the next policy
            policy = self.get_policy(root, temperature)
            
            # Decide the next move based on the policy
            move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=policy)
            
            # Play the move
            game.play(move)
        
            # Log status to the console
            self.console_print(game_nr, root, game, policy)

            # Update the root node
            root = root.children[move]
            
            # Store the intermediate state
            states.append((encoded_board, policy))
            
            # If somebody won
            if game.won():

                if (game.player * -1) == -1:
                    print('X won \n')
                    value = -1
                if (game.player * -1) == 1:
                    print('O won \n')
                    value = 1
                    
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

        return np.array(states_values)

    def console_print(self, game_nr, root, game, policy):
        """
        Log status to the console
        """
        # Predict the policy and value of the board state
        policy_estimate, value_estimate = self.net.model.predict(np.array([game.encoded()]))

        policy_print      = ['{:.2f}'.format(value) for value in policy]
        policy_pred_print = ['{:.2f}'.format(value) for value in policy_estimate[0]]

        print(f'Game: {game_nr}', 'Move:', '\033[95mX\033[0m (-1)' if game.player == 1 else '\033[92mO\033[0m (1)', '\n')

        print('Policy:     ', ' | '.join(policy_print))
        print('Policy pred:', ' | '.join(policy_pred_print), '\n')

        print('Q Value:   ', int(round(root.total_value / root.parent.child_number_visits[root.move])))
        print('Value pred:', round(value_estimate[0][0], 2), '\n')

        game.presentation()
        print()

    def replay(self, iteration):
        """
        Train the model by replaying from memory
        """
        for _ in range(self.training_loops):
            # Sample a minibatch
            boards, policies, values = self.memory.get_minibatch(self.minibatch_size)

            # Train the model
            self.net.model.fit(boards, [policies, values], batch_size=self.minibatch_size, shuffle=False, epochs=1)

        self.net.save(iteration)

    def run(self):
        """
        Run the simulation
        """
        # Start at the latest game
        game_nr = self.memory.latest_game

        while True:

            with Pool(5) as p:
                print(p.map(self.self_play, [1, 2, 3]))

            # # Run one game of self play
            # game = self.self_play(game_nr)

            # # Store the game in memory
            # self.memory.remember(game, game_nr)

            # Train the model
            if (game_nr % self.games_per_iteration == 0) and (self.memory.filled):
                iteration = int(game_nr / self.games_per_iteration)
                self.replay(iteration)

            game_nr += 1