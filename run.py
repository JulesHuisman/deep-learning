import numpy as np

from connect_net import ConnectNet
from simulation import Simulation

if __name__ == '__main__':
    # The neural network
    net = ConnectNet('ConnectNet-V1')
    # net.load(6)

    best_net = ConnectNet('ConnectNet-V1')
    # best_net.load(6)

    # The simulation environment
    simulation = Simulation(net=net,
                            best_net=best_net,
                            games_per_iteration=40,
                            moves_per_game=50,
                            memory_size=5000,
                            minibatch_size=256,
                            training_loops=20)

    # Start the simulation
    simulation.run()