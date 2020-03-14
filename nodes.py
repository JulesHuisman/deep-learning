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
        self.expanded = False
        
        # Parent of the node
        self.parent = parent
        
        # Information of the children of this node
        self.children = {}
        self.child_priors = np.zeros([7], dtype=np.float32)
        self.child_total_value = np.zeros([7], dtype=np.float32)
        self.child_number_visits = np.zeros([7], dtype=np.float32)
        
        # Possible moves to take from this node
        self.legal_moves = []
        
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
        """
        The Q value of the children
        value / number of visits -> average value
        """
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        """The U value of the children"""
        return sqrt(self.number_visits) * (abs(self.child_priors) / (1 + self.child_number_visits))
    
    def best_child(self):
        """
        Returns the best child based on the Q and U value (exploration vs exploitation)
        """
        if self.legal_moves != []:
            uct_values = self.child_Q() + self.child_U()
            best_child = self.legal_moves[np.argmax(uct_values[self.legal_moves])]
        else:
            best_child = np.argmax(self.child_Q() + self.child_U())
            
        return best_child
    
    def select_leaf(self):
        """
        Traverse the tree by expanded nodes
        """
        current = self
        
        while current.expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
            
        return current
    
    def add_dirichlet_noise(self, next_moves, child_priors):
        """
        Add noise to the child priors.
        This adds to randomness to the process
        """
        valid_child_priors = child_priors[next_moves]
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(np.zeros([len(valid_child_priors)], dtype=np.float32) + 192)
        child_priors[next_moves] = valid_child_priors
        
        return child_priors
    
    def expand(self, child_priors):
        """
        MCTS expand action.
        This adds the children to the current node.
        """
        # Set expanded flag to true
        self.expanded = True
        
        # Legal moves
        self.legal_moves = self.game.moves()
        self.child_priors = child_priors
        
        # No possible actions (node is a leaf)
        if self.legal_moves == []:
            self.expanded = False
        
        # Mask all illegal actions
        self.child_priors[[i for i in range(len(self.child_priors)) if i not in self.legal_moves]] = 0.000000000
        
        # Add dirichlet noise to priors in root node
        if not self.parent.parent:
            self.child_priors = self.add_dirichlet_noise(self.legal_moves, self.child_priors)
    
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
        Go back up the tree and increase visits and adjust value of all parent nodes.
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