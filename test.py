from game import Game
from connect_net import ConnectNet
from nodes import DummyNode, Node
from simulation import Simulation
from memory import Memory
import numpy as np

MODEL = 'DeepFour-V1'

# np.set_printoptions(precision=3)

# memory = Memory(folder=f'data/{MODEL}/memory', size=80000)
# memory.load_memories()

# print('Memory size:', len(memory.memory))

# values = [value for board, policy, value in memory.memory]
# print('Average value:', np.mean(values))

# for training in range(10):
#     boards, policies, values = memory.get_minibatch(1024)
#     print(np.mean(policies, 0))

# zero = ConnectNet(MODEL)
# # zero.load(0)

# for training in range(10):
#     # Sample a minibatch
#     boards, policies, values = memory.get_minibatch(1024)

#     print('Average value:', np.mean(values))

#     # Train the model
#     zero.model.fit(boards, [policies, values], batch_size=32, shuffle=False, epochs=1)

# zero.save('current')

# The simulation environment
simulation = Simulation(net_name=MODEL,
                        games_per_iteration=64,
                        moves_per_game=512,
                        memory_size=80000,
                        minibatch_size=256,
                        training_loops=10,
                        workers=16,
                        duel_threshold=0.60)

Simulation.self_play(simulation, 0)
# Simulation.dual(simulation)