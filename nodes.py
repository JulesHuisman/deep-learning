from collections import defaultdict
from copy import deepcopy
from math import sqrt
import numpy as np

class DummyNode:
    """
    Empty node, used as parent of the root node.
    """
    def __init__(self):
        self.parent = None
        self.child_total_value = defaultdict(float)
        self.child_number_visits = defaultdict(float)

class Node:
    """
    One nodes represents one board state.
    Used for Monte Carlo Tree Search.
    https://github.com/plkmo/AlphaZero_Connect4/blob/master/src/MCTS_c4.py
    """
    def __init__(self, game, move, parent=None):
        # Game state
        self.game = game
        
        # Index of the move (1-7)
        self.move = move
        
        # Has this node been expanded
        self.is_expanded = False
        
        # Parent of the node
        self.parent = parent
        
        # Information of the children of this node
        self.children = {}
        self.child_priors = np.zeros([7], dtype=np.float32)
        self.child_total_value = np.zeros([7], dtype=np.float32)
        self.child_number_visits = np.zeros([7], dtype=np.float32)
        
        # Possible actions to take
        self.action_indices = []
        
    @property
    def number_visits(self):
        """How many times has this node been visited"""
        return self.parent.child_number_visits[self.move]
    
    @number_visits.setter
    def number_visits(self, value):
        """Set the number of visits (stored in the parent node)"""
        self.parent.child_number_visits[self.move] = value
        
    @property
    def total_value(self):
        """Value of this node"""
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        """Set the value of this node"""
        self.parent.child_total_value[self.move] = value
    
    def child_Q(self):
        """The Q value of the children"""
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        """The U value of the children"""
        return sqrt(self.number_visits) * (abs(self.child_priors) / (1 + self.child_number_visits))
    
    def best_child(self):
        """
        Returns the best child based on the Q and U value (exploration vs exploitation)
        """
        if self.action_indices != []:
            best_move = self.child_Q() + self.child_U()
            best_move = self.action_indices[np.argmax(best_move[self.action_indices])]
        else:
            best_move = np.argmax(self.child_Q() + self.child_U())
            
        return best_move
    
    def select_leaf(self):
        """
        Traverse the tree by expanded nodes
        """
        current = self
        
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
            
        return current
    
    def add_dirichlet_noise(self, action_idices, child_priors):
        """
        Add noise to the child priors.
        This adds to randomness to the process
        """
        valid_child_priors = child_priors[action_idices]
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(np.zeros([len(valid_child_priors)], dtype=np.float32) + 192)
        child_priors[action_idices] = valid_child_priors
        
        return child_priors
    
    def expand(self, child_priors):
        """
        MCTS expand action.
        This adds the children to the current node.
        """
        # Set expanded flag to true
        self.is_expanded = True
        
        # Legal moves
        action_indices = self.game.moves()
        c_p = child_priors
        
        # No possible actions (node is a leaf)
        if action_indices == []:
            self.is_expanded = False
            
        self.action_indices = action_indices
        
        # Mask all illegal actions REVISIT
        c_p[[i for i in range(len(child_priors)) if i not in action_indices]] = 0.000000000
        
        # Add dirichlet noise to priors in root node
        if not self.parent.parent:
            c_p = self.add_dirichlet_noise(action_indices, c_p)
            
        self.child_priors = c_p
    
    def maybe_add_child(self, move):
        """
        Try to add a child to the current node if it doesn't exist yet
        """
        if move not in self.children:
            # Create a deepcopy of the game
            game = deepcopy(self.game)
            
            # Play the move
            game.play(move)
            
            # Add the child
            self.children[move] = Node(game, move, parent=self)
            
        return self.children[move]
    
    def backup(self, value_estimate):
        """
        MCTS backup action.
        FILL
        """
        current = self
        
        # While not back at the root
        while current.parent is not None:
            # Add one visit to the node
            current.number_visits += 1
            
            if current.game.player == -1:
                current.total_value += (1 * value_estimate)
            elif current.game.player == 1:
                current.total_value += (-1 * value_estimate)
                
            # Traverse up the tree
            current = current.parent

def print_tree(node, _prefix="", _last=True):
    print(_prefix, "`- " if _last else "|- ", node.move, ' | ', '{:.2f}'.format(node.total_value), sep="")

    _prefix += "   " if _last else "|  "

    for child in node.children.values():
        _last = len(child.children) == 0
        print_tree(child, _prefix, _last)