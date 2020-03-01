import pandas as pd
import numpy as np

from utils import *
from tqdm import tqdm, tqdm_notebook

if __name__ != '__main__':
    tqdm = tqdm_notebook

class Simulation:
    def __init__(self, stocks, agent, env):
        self.agent = agent
        self.env   = env

        # Register the agent to the environment
        env.register(agent)

    def run(self, episodes=10):
        with tqdm(total=episodes) as episodes_progress:
            for episode in range(episodes):
                state, position = self.env.reset()
                done = False

                with tqdm(total=self.env.number_of_steps) as steps_progress:
                    while not done:
                        next_position = self.agent.act(state, position)
                        next_state, next_position, done = self.env.step(next_position)

                        next_position = position
                        next_state = state

                        steps_progress.update(1)

                    # Train the agent
                    self.agent.train()

                    # Store the data of the episode
                    self.agent.save_logs(episode=episode)

                    # Print a summary of the run
                    self.agent.print_summary()

                    episode_info = self.agent.summary

                    episodes_progress.set_postfix(**episode_info)

                    steps_progress.reset()
                    episodes_progress.update(1)