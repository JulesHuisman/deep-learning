import random
import multiprocessing
import numpy as np

from simulation import Simulation

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat
from shutil import copyfile


if __name__ == '__main__':

    # The simulation environment
    simulation = Simulation(net_name='DeepFour-V1',
                            games_per_iteration=128,
                            moves_per_game=256,
                            memory_size=80000,
                            minibatch_size=512,
                            training_loops=10,
                            workers=16,
                            duel_threshold=0.60)

    # Create a pool of workers and execute the duels
    with ProcessPoolExecutor(max_workers=simulation.workers) as executor:
        duels = [executor.submit(simulation.duel) for _ in range(16)]

        results = np.array([duel.result() for duel in duels])
        no_draws = results[results != 'draw']
        winrate = len(no_draws[no_draws == 'checkpoint']) / len(no_draws)

        print('Final results:', results)
        print('Final results no draws:', no_draws)
        print('Win rate:', winrate)

        with open('duels.txt', 'a+') as file:
            file.write(str(winrate) + '\n')
            file.write(str(' '.join(results)) + '\n')