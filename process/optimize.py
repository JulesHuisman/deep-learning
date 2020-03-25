import os

from deepfour import DeepFour
from time import sleep

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

class OptimizeProcess:
    """
    https://github.com/Zeta36/connect4-alpha-zero
    """
    def __init__(self, config, memory):
        self.config = config
        self.memory = memory
        self.checkpoint = DeepFour(config)
        self.checkpoint.load('checkpoint')

    def optimize(self):
        checkpoint_steps = 0
        total_steps = self.last_version() * self.config.checkpoint_steps
        update_steps = total_steps % self.config.update_steps
        self.memory.load_memories()

        while True:
            print(f'Total steps: {total_steps} | Checkpoint steps: {checkpoint_steps} | Update steps: {update_steps} | Memory size: {len(self.memory.memory)}')

            # Not enough data yet
            if len(self.memory.memory) < 50000:
                print(f'Not enough data yet ({len(self.memory.memory)})')
                sleep(60)
                self.memory.load_memories()
                continue

            self.train()

            total_steps += 1
            checkpoint_steps += 1
            update_steps += 1

            # Checkpoint the model every n steps
            if checkpoint_steps >= self.config.checkpoint_steps:
                self.checkpoint.save(self.last_version() + 1)
                self.checkpoint.save('checkpoint')
                self.memory.load_memories()
                checkpoint_steps = 0

            # Create a new challenger every n steps
            if update_steps >= self.config.update_steps:
                print('New challenger')
                self.checkpoint.save('challenger')
                update_steps = 0

    def train(self):
        """
        Train the model on one minibatch
        """
        boards, policies, values = self.memory.get_minibatch()

        self.checkpoint.model.fit(boards, [policies, values], batch_size=self.config.batch_size, shuffle=False, epochs=1)

    def last_version(self):
        """
        Latest model version
        """
        return sorted([int(modelname.split('.')[-2]) for modelname in self.historic_models()])[-1]

    def historic_models(self):
        """
        Load model iterations
        """
        model_folder = os.path.join('data', self.config.model, 'models')

        return [modelname for modelname in os.listdir(model_folder) if is_int(modelname.split('.')[-2])]