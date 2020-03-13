from mcts import self_play
from connect_net import ConnectNet

if __name__ == '__main__':
    # The neural network
    net = ConnectNet().model

    # Start self play
    self_play(net, 10)