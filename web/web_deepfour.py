import sys
import os
import numpy as np

os.environ['KMP_WARNINGS'] = 'off'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.realpath('..'))

from deepfour import DeepFour
from simulation.mcts import mcts, get_policy
from simulation.nodes import StateNode

class Config:
    def __init__(self):
        self.model = 'DeepFour'
        self.n_filters = 64
        self.kernel = 4
        self.value_dense = 32
        self.res_layers = 5
        self.l2_reg = 0.0001

class WebDeepFour(DeepFour):
    def load(self):
        """Load model weights"""
        try:
            self.model.load_weights('model.h5')
            self.model._make_predict_function()
            print('Loaded network!')
        except:
            print('Failed to load network!')

def move(game, net, set_session, sess, graph):
    with graph.as_default():
        set_session(sess)

        game.player = -1

        root = StateNode(game=game,
                         c_puct=1.5,
                         depth=0,
                         move=move)

        root = mcts(root, 700, net)

        return np.argmax(root.child_number_visits)