import numpy as np
import sys
import os

sys.path.append(os.path.realpath('..'))

from simulation.mcts import mcts, get_policy
from deepfour import DeepFour
from simulation.nodes import StateNode

def move(game, net):
    game.player = -1

    root = StateNode(game=game,
                     c_puct=1.5,
                     depth=0,
                     move=move)

    root = mcts(root, 700, net)

    return np.argmax(root.child_number_visits)
