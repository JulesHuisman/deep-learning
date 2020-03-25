from simulation.mcts import mcts, get_policy
from simulation.game import Game
from deepfour import DeepFour
from simulation.nodes import StateNode
from config import Config

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Play a game against the AI')
parser.add_argument('--version', metavar='version', type=str, default='checkpoint', help='Version of the AI to play against')

np.set_printoptions(precision=2, suppress=True, linewidth=150)

args = parser.parse_args()

class Play:
    def __init__(self, game, net):
        self.game = game
        self.net  = net

    def start(self):
        done = False
        move = None

        while not done:
            print()
            print('Turn:', '\033[95mAI\033[0m' if self.game.player == -1 else '\033[92mYou\033[0m')
            print()
            print(' ', ' '.join([f" {col} " for col in np.arange(1, 8)]))

            self.game.presentation()
            print()

            root = StateNode(game=self.game,
                             move=move)

            # AI turn
            if self.game.player == -1:

                # Run MCTS
                root = mcts(root, 512, self.net)

                # Move to the node with the most visits
                move = np.argmax(root.child_number_visits)
            else:
                while True:
                    try:
                        move = int(input('Position? ')) - 1
                    except KeyboardInterrupt:
                        return
                    except:
                        continue
                    if move in self.game.moves():
                        break

            # Play the stone
            self.game.play(move)

            # Game was won
            if self.game.won():
                self.game.presentation()
                if self.game.player * -1 == 1:
                    print('\n\033[92mYou won!\033[0m \n')
                else:
                    print('\n\033[95mAI won!\033[0m \n')
                done = True
            elif self.game.moves == []:
                print('\nDraw')
                done = True

if __name__ == '__main__':
    game = Game()
    net = DeepFour(Config())
    net.load(args.version)
    
    play = Play(game, net)
    play.start()