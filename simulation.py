import pandas as pd
import numpy as np

from utils import *
from tqdm import tqdm, tqdm_notebook

if __name__ != '__main__':
    tqdm = tqdm_notebook

class Simulation:
    def __init__(self, agent, train_env, test_env):
        self.agent     = agent
        self.train_env = train_env
        self.test_env  = test_env
        # self.logger    = Logger()

        # Register the agent to the environments
        self.train_env.register(agent, training=True)
        self.test_env.register(agent, training=False)

    def run(self, episodes):
        """
        Run the different episodes
        """
        with tqdm(total=episodes) as episodes_progress:
            with tqdm(total=(self.train_env.number_of_steps + self.test_env.number_of_steps)) as steps_progress:

                for episode in range(episodes):
                    self.run_episode(env=self.train_env,
                                     episode=episode,
                                     training=True,
                                     steps_progress=steps_progress)

                    # Test every 5 episodes
                    if episode % 5 == 0:
                        self.run_episode(env=self.test_env,
                                         episode=episode,
                                         training=False,
                                         steps_progress=steps_progress)

                    steps_progress.reset()
                    episodes_progress.update(1)

    def run_episode(self, env, episode, training, steps_progress):
        """
        Run a single episode, can be training or testing
        """
        self.agent.training = training

        state, position = env.reset()
        done = False

        while not done:
            next_position, q_value = self.agent.act(state, position)
            next_state, next_position, done = env.step(next_position, q_value)

            position = next_position
            state = next_state

            steps_progress.update(1)

        # Store the data of the episode
        if episode % 5 == 0:
            self.agent.save_logs(episode=episode)

        # Tensorboard log
        env.tensorboard(episode=episode)