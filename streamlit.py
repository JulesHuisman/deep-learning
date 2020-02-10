import streamlit as st
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

from collections import deque
from metaflow import Flow
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

BUY = 0
SELL = 1
HOLD = 2

@st.cache
def split_sequence(sequence, n_steps):
    X = []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        seq_x = sequence[i:end_ix]
        X.append(seq_x)
        
    return np.array(X)

def plot(data, title, xlabel, ylabel):
    fig = px.line(
        x=list(range(len(data))), 
        y=data,
        title=title)

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)

    st.plotly_chart(fig) 

@st.cache
def get_dataframe():
    run = Flow('TraderFlow').latest_successful_run
    return run.data.df

df = get_dataframe()

stock = st.sidebar.selectbox(
    'Which stock to train on?', 
    df.columns)

episodes = st.sidebar.slider(
    'Number of episodes', 
    0, 300,
    value=10)

window_size = st.sidebar.slider(
    'Window size', 
    1, 21,
    value=14)

sample_batch_size = st.sidebar.slider(
    'Experience replay batch size', 
    8, 64,
    value=32)

train_data = split_sequence(df[stock].dropna(), window_size)

class Trader:
    def __init__(self, 
                 discount_rate=0.95, 
                 exploration_rate=1.,
                 exploration_decay=.9,
                 state_size=window_size,
                 action_size=3,
                 learning_rate=0.001):
        
        self.discount_rate     = discount_rate
        self.exploration_rate  = exploration_rate
        self.exploration_decay = exploration_decay
        self.state_size        = state_size
        self.action_size       = action_size
        self.memory            = deque(maxlen=1000)
        self.learning_rate     = learning_rate
        
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def get_next_action(self, state):
        if random.random() > self.exploration_rate:
            return self.model_action(state)
        else:
            return self.random_action()
        
    def model_action(self, state):
        return np.argmax(self.model.predict(state)[0])
    
    def random_action(self):
        return random.randrange(self.action_size)
    
    def remember(self, state, next_state, action, reward):
        self.memory.append((state, next_state, action, reward))
        
    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        
        sample_batch = random.sample(self.memory, sample_batch_size)
        
        for (state, next_state, action, reward) in sample_batch:
            model_pred = self.model.predict(next_state)
            
            # Fixes a bug where the model doesn't predict
            if len(model_pred) == 0:
                continue
            
            target = reward + (self.discount_rate * np.amax(model_pred[0]))
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        self.update_rate()
            
    def update_rate(self):
        # Move the exploration rate to zero to stop exploring
        self.exploration_rate *= self.exploration_decay

class Environment:
    def __init__(self, 
                 data,
                 progress,
                 state_size=window_size,
                 episodes=3,
                 sample_batch_size=64):
        self.logs              = []
        self.profits           = 0
        self.data              = data
        self.stocks            = []
        self.episodes          = episodes
        self.state_size        = state_size
        self.sample_batch_size = sample_batch_size
        self.progress          = progress
        
        self.trader = Trader(state_size=self.state_size)
        
    def reset(self):
        self.profits = 0
        self.stocks  = []
        
    def buy_stocks(self):
        self.stocks.append(self.stock_price)
        return 0
        
    def sell_stocks(self):
        # If there is no inventory
        if len(self.stocks) == 0:
            return 0
        
        bought_price = self.stocks.pop(0)
        profit = self.stock_price - bought_price
        self.profits += profit
        
        reward = profit
        
        return reward
    
    def run(self):
        for episode in range(self.episodes):
            
            state        = self.data[0:1]
            reward       = 0
            done         = False
            index        = 0
            cons_buys    = 0
            cons_sells   = 0
            cons_holds   = 0
            logs         = {'profit': [], 'actions': []}
            
            while not done:
                index += 1
                
                action = self.trader.get_next_action(state)
                self.stock_price = state[0][-1]
                
                if action == BUY:
                    reward = self.buy_stocks()
                    cons_buys += 1
                    cons_sells = 0
                    cons_holds = 0
                elif action == SELL:
                    reward = self.sell_stocks() * 100
                    cons_sells += 1
                    cons_buys = 0
                    cons_holds = 0
                elif action == HOLD:
                    reward = 0
                    cons_holds += 1
                    cons_buys = 0
                    cons_sells = 0
                    
                if (cons_buys > 10) or (cons_sells > 10) or (cons_holds > 20):
                    reward -= 200

                next_state = self.data[index:index+1]

                self.trader.remember(state, next_state, action, reward)
                state = next_state
                
                logs['profit'].append(self.profits)
                logs['actions'].append(action)
        
                if (index >= len(self.data)):
                    done = True
    
            print(f'Episode: {episode} Profit: {int(self.profits)}')

            self.trader.replay(self.sample_batch_size)
            self.reset()
            self.progress.progress(int((episode + 1) / self.episodes * 100))
            self.logs.append(logs)

            # final_profits = [log['profit'][-1] for log in self.logs]

            # plot(final_profits, 'Profits', 'Episode', 'Profit')

st.title('Deep Q Trader')

st.header('Stock data')
st.dataframe(df.tail())
st.header(f'Train data of {stock}')
st.write(train_data[:3])

st.header('Run model')
progress = st.progress(0)

if st.button('Run'):
    env = Environment(
        progress=progress,
        data=train_data,
        episodes=episodes,
        sample_batch_size=sample_batch_size)

    env.run()

    with open('logs.pkl', 'wb') as handle:
        pickle.dump(env.logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('logs.pkl', 'rb') as handle:
    logs = pickle.load(handle)

    final_profits = [log['profit'][-1] for log in logs]

    plot(final_profits, 'Profits', 'Episode', 'Profit')

    episode_log = st.slider(
        'Episode run', 
        0, len(logs)-1,
        value=0)

    plot(logs[episode_log]['profit'], 'Profits', 'Timestamp', 'Profit')