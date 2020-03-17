from simulation import Simulation

if __name__ == '__main__':

    # The simulation environment
    simulation = Simulation(net_name='DeepFour-V1',
                            games_per_iteration=75,
                            moves_per_game=150,
                            memory_size=90000,
                            minibatch_size=256,
                            training_loops=10,
                            workers=5,
                            games_per_pool=25)

    # Start the simulation
    simulation.run()