import numpy as np
import os

from random import sample
from collections import deque

class Memory:
    def __init__(self, folder, size):
        self.folder = folder
        self.size = size
        self.memory = deque(maxlen=size)

        # Create a storage folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Load memories from file system
        self.load_memories()

    @property
    def filled(self):
        """Is the memory filled up?"""
        return len(self.memory) == self.memory.maxlen

    @property
    def sorted_games(self):
        """Return game memories in order"""
        return sorted(os.listdir(self.folder), reverse=True, key=lambda filename: int(filename.replace('-', '.').split('.')[1]))

    @property
    def latest_game(self):
        """Number of the latest stored game"""
        if self.sorted_games != []:
            return int(self.sorted_games[0].replace('-', '.').split('.')[1])
        else:
            return 0

    def remember(self, game, game_nr):
        """
        Store a game in memory and in file storage
        """
        # Store in file storage
        np.save(os.path.join(self.folder, f'game-{game_nr}.npy'), game)

        # Store in memory
        for move in game:
            self.memory.append(move)

    def load_memories(self):
        """
        Load memories from file storage.
        Fill it up until limit reached
        """
        for filename in self.sorted_games:
            if filename.endswith('.npy'):
                game = np.load(os.path.join(self.folder, filename), allow_pickle=True)
                
                for move in game:
                    self.memory.appendleft(move)

                if self.filled:
                    break

        print('Memory size:', len(self.memory), '\n')

    def get_minibatch(self, size):
        """
        Get a random minibatch from memory
        """
        minibatch = sample(self.memory, size)

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