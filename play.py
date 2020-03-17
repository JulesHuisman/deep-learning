from simulation import Simulation
from game import Game
from connect_net import ConnectNet
from nodes import DummyNode, Node

import numpy as np

class Play:
    def __init__(self, game, net):
        self.game = game
        self.net  = net

    def start(self):
        done = False

        move = None

        root = DummyNode()

        while not done:
            print()
            print('Turn:', '\033[95mAI\033[0m' if self.game.player == -1 else '\033[92mYou\033[0m')
            print()
            print(' ', ' '.join([f" {col} " for col in np.arange(1, 8)]))

            self.game.presentation()
            print()

            # AI turn
            if self.game.player == -1:
                if move in root.children:
                    root = root.children[move]
                else:
                    root = Node(game=self.game,
                                move=move,
                                parent=root)

                # Run MCTS
                root = Simulation.mcts(root, 1000, self.net.model)

                # Get the playing policy based on the child visits
                policy = Simulation.get_policy(root, 1)

                print(policy)

                move = np.argmax(policy)
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
    net = ConnectNet('DeepFour-V1')
    net.load('current')
    
    play = Play(game, net)
    play.start()