import numpy as np
import os

from random import sample
from collections import deque

class Memory:
    def __init__(self, config):
        self.config = config

        # Memory is a double ended queue to quickly append items
        self.memory = deque(maxlen=self.config.memory_size)

        # The storage location
        self.folder = os.path.join('data', config.model, 'memory')

        # Create a storage folder (if does not exist)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    @property
    def filled(self):
        """Is the memory filled up?"""
        return len(self.memory) == self.memory.maxlen

    @property
    def sorted_games(self):
        """Return game memories in order"""
        return sorted(os.listdir(self.folder), reverse=True, key=lambda filename: int(filename.replace('-', '.').split('.')[1]))

    def n_games(self):
        """Returns the number of games played"""
        return len([_ for _ in os.listdir(self.folder)])

    def remember(self, game, identifier):
        """
        Store a game in file storage
        """
        # Tranform to numpy array
        game = np.array(game)

        # Store in file storage
        np.save(os.path.join(self.folder, f'game-{identifier}.npy'), game)

    def load_memories(self):
        """
        Load memories from file storage.
        Fill it up until limit reached
        """
        self.memory.clear()

        for filename in self.sorted_games:
            if filename.endswith('.npy'):
                try:
                    game = np.load(os.path.join(self.folder, filename), allow_pickle=True)
                    
                    for move in game[::-1]:
                        self.memory.appendleft(move)

                    if self.filled:
                        break
                except:
                    continue

    def get_minibatch(self):
        """
        Get a random minibatch from memory
        """
        minibatch = sample(self.memory, self.config.batch_size)

        boards   = []
        policies = []
        values   = []

        for board, policy, value in minibatch:
            boards.append(board)
            policies.append(policy)
            values.append(value)
            
        boards   = np.array(boards)
        policies = np.array(policies)
        values   = np.array(values)

        return boards, policies, values