import numpy as np

from nodes import DummyNode, Node
from game import Game
from copy import deepcopy

def search(game, num_reads, net):
    """
    Perform one Monte Carlo Tree Search.
    Decides the next play.
    """

    # The root node of the current search
    root = Node(game=game,
                move=None,
                parent=DummyNode())
    
    for i in range(num_reads):
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

def get_policy(root, temp=1):
    """
    Policy is based on the number of visits to that node.
    Better nodes get visit more often.
    Normalize the value of the moves.
    """
    n_visits = root.child_number_visits ** (1 / temp)
    return n_visits / sum(n_visits)

def self_play(net, num_games):
    # Final training dataset (encoded board state, policy, value)
    dataset = []
        
    for game_nr in range(num_games):
        print(f'Simulating game {game_nr + 1} \n')
        
        # Create a new empty game
        game = Game()
        
        # Storage for states and policies
        states = []
        
        value = 0
        move_count = 0
        done = False
        
        # Keep playing while the game is not done
        while not done and game.moves() != []:
            
            # FILL
            if move_count < 11:
                t = 1
            else:
                t = 0.1
            
            # Encoded board
            encoded_board = deepcopy(game.encoded())
            
            # Run UCT search
            root = search(game, 700, net)
            
            # Get the next policy
            policy = get_policy(root, t)
            
            # Decide the next move based on the policy
            move = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=policy)
            
            game.play(move)
            
            print('Policy: {:.2f}'.format(policy))
            print(game.presentation(), '\n')
            
            states.append((encoded_board, policy))
            
            # If somebody won (lag of one)
            if game.won():
                
                if game.player == 1:
                    print('X won \n')
                    value = -1
                if game.player == -1:
                    print('O won \n')
                    value = 1
                    
                done = True
            
            # Increase the total moves
            move_count += 1
        
        for index, data in enumerate(states):
            if index == 0:
                dataset.append((data[0], data[1], 0))
            else:
                dataset.append((data[0], data[1], value))
                
    return dataset