import numpy as np

from connect_net import ConnectNet
from simulation import Simulation

if __name__ == '__main__':
    # The neural network
    net = ConnectNet('ConnectNet-V1')
    net.load(3)

    # The simulation environment
    simulation = Simulation(net=net,
                            games_per_iteration=40,
                            moves_per_game=500,
                            memory_size=32000,
                            minibatch_size=256,
                            training_loops=10)

    # Start the simulation
    simulation.run()