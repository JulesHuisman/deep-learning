import os
import numpy as np
import mlflow

from deepfour import DeepFour
from time import sleep
from process.evaluate import EvaluateProcess

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
        self.evaluate = EvaluateProcess(self.config)

    def optimize(self):
        checkpoint_steps = 0
        total_steps = self.last_version() * self.config.checkpoint_steps
        update_steps = total_steps % self.config.update_steps
        self.memory.load_memories()
        self.evaluate.evaluate(loop=False)

        while True:
            print(f'Total steps: {total_steps} | Checkpoint steps: {checkpoint_steps} | Update steps: {update_steps} | Memory size: {len(self.memory.memory)}')

            # Not enough data yet
            # if len(self.memory.memory) < self.config.memory_size:
            if len(self.memory.memory) < 50000:
                print(f'Not enough data yet ({len(self.memory.memory)})')
                sleep(60)
                self.memory.load_memories()
                continue

            self.train(total_steps)

            total_steps      += 1
            checkpoint_steps += 1
            update_steps     += 1

            if total_steps % 100 == 0:
                self.memory.load_memories()

            # Checkpoint the model every n steps
            if checkpoint_steps >= self.config.checkpoint_steps:
                self.checkpoint.save(self.last_version() + 1)
                self.checkpoint.save('checkpoint')

                # Apply the learning rate schedule
                # self.checkpoint.update_lr(total_steps)

                checkpoint_steps = 0

            # Create a new challenger every n steps
            if update_steps >= self.config.update_steps:
                self.checkpoint.save('challenger')
                self.evaluate.evaluate(loop=False)
                update_steps = 0

    def train(self, total_steps):
        """
        Train the model on one minibatch
        """
        boards, policies, values = self.memory.get_minibatch()

        history = self.checkpoint.model.fit(boards, [policies, values], batch_size=self.config.batch_size, shuffle=False, epochs=1)

        if total_steps % 100 == 0:
            # Log to mlflow
            mlflow.log_metric('loss', np.mean(history.history['loss']), total_steps)
            mlflow.log_metric('policy-loss', np.mean(history.history['policy_loss']), total_steps)
            mlflow.log_metric('value-loss', np.mean(history.history['value_loss']), total_steps)
            mlflow.log_metric('value-mean', np.mean(history.history['value_mean']), total_steps)

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