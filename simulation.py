import pandas as pd
import numpy as np

from utils import *

class Simulation:
    def __init__(self, stocks, agent, env):
        self.prices = stocks.prices
        self.agent  = agent
        self.env    = env

    def run(self, episodes=10):
        for episode in range(episodes):
            print('Episode', episode)

            state = self.env.reset()
            done = False

            while not done:
                action, position = self.agent.act(state)
                next_state, reward, done = self.env.step(action, position)

                next_state = state

            # Store the data of the episode
            self.agent.save_logs(episode=episode)