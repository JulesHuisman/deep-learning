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
                            games_per_iteration=32,
                            moves_per_game=256,
                            memory_size=100_000,
                            minibatch_size=1024,
                            training_loops=10,
                            workers=16,
                            duel_threshold=0.60)

    while True:
        for _ in range(4):
            # Start after the latest recorded game
            start_nr = (simulation.memory.latest_game + 1)

            # Create a queue of self plays to run
            self_plays = zip(repeat(Simulation.self_play),
                             repeat(simulation),
                             [start_nr + game_nr for game_nr in range(simulation.games_per_iteration)])

            # Create a pool of workers and execute the self plays
            with ProcessPoolExecutor(max_workers=simulation.workers) as executor:
                results = [executor.submit(*self_play) for self_play in self_plays]

        # Train the model by replaying from memory (needs to be isolated from the self play workers)
        with ProcessPoolExecutor(max_workers=1) as executor:
            executor.submit(simulation.replay)

        # Create a pool of workers and execute the duels
        with ProcessPoolExecutor(max_workers=simulation.workers) as executor:
            duels = [executor.submit(simulation.duel) for _ in range(32)]

            results = np.array([duel.result() for duel in duels])
            no_draws = results[results != 'draw']
            winrate = len(no_draws[no_draws == 'checkpoint']) / len(no_draws)

            print('Final results:', results)
            print('Final results no draws:', no_draws)
            print('Win rate:', winrate)

            with open('duels.txt', 'a+') as file:
                file.write(str(winrate) + '\n')
                file.write(str(' '.join(results)) + '\n')

            # If the checkpoint model is better than the previous best, update the best
            if winrate >= simulation.duel_threshold:
                print('Updating best')
                copyfile(f'data/{simulation.net_name}/models/{simulation.net_name}-checkpoint.h5', f'data/{simulation.net_name}/models/{simulation.net_name}-best.h5')