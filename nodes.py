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
        self.children = {}
        self.child_total_value = defaultdict(float)
        self.child_number_visits = defaultdict(float)

class Node:
    """
    One nodes represents one board state.
    Used for Monte Carlo Tree Search.
    https://github.com/plkmo/AlphaZero_Connect4/blob/master/src/MCTS_c4.py
    """
    def __init__(self, game, move=None, depth=0, parent=DummyNode()):
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

        # Keep track of the depth of this game state
        self.depth = depth
        
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
    
    def expand(self, child_priors):
        """
        Expand the current node, add the child priors from the neural network.
        """
        # Set expanded flag to true
        self.expanded = True
        
        # Legal moves
        self.legal_moves = self.game.moves()
        self.child_priors = child_priors
        
        # No possible actions (node is a final state)
        if self.legal_moves == []:
            self.expanded = False
        
        # Mask all illegal actions
        self.child_priors[[i for i in range(len(self.child_priors)) if i not in self.legal_moves]] = 0.000000000
    
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
            self.children[move] = Node(game, move, parent=self, depth=(self.depth + 1))
            
        return self.children[move]
    
    def backprop(self, value_estimate):
        """
        Go back up the tree and increase visits and adjust value of all parent nodes.
        """
        current = self

        # Go back up the tree until back at the root
        while current.parent is not None:
            # Add one visit to the node
            current.number_visits += 1
            
            if current.game.player == 1:
                current.total_value += (1 * value_estimate)  # value estimate +1 = O wins
            elif current.game.player == -1:
                current.total_value += (-1 * value_estimate)  # value estimate -1 = O wins
                
            # Back up the tree
            current = current.parent