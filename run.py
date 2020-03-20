import random
import multiprocessing

from simulation import Simulation

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

if __name__ == '__main__':

    # The simulation environment
    simulation = Simulation(net_name='DeepFour-V1',
                            games_per_iteration=64,
                            moves_per_game=256,
                            memory_size=80000,
                            minibatch_size=512,
                            training_loops=10,
                            workers=16)

    while True:
        # Start after the latest recorded game
        start_nr = (simulation.memory.latest_game + 1)

        # Create a queue of self plays to run
        self_plays = zip(repeat(Simulation.self_play),
                         repeat(simulation),
                         [start_nr + game_nr for game_nr in range(simulation.games_per_iteration)])

        # Create a pool of workers and execute the self plays
        with ProcessPoolExecutor(max_workers=simulation.workers) as executor:
            results = [executor.submit(*self_play) for self_play in self_plays]

        # # Train the model by replaying from memory (needs to be isolated from the self play workers)
        # with ProcessPoolExecutor(max_workers=1) as executor:
        #     executor.submit(simulation.replay)

        # # Create a pool of workers and execute the duals
        # with ProcessPoolExecutor(max_workers=simulation.workers) as executor:
        #     results = [executor.submit((simulation.duel,)).result() for _ in range(16)]

        #     print(results)

        # Dual against best