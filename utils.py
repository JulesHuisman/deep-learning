import numpy as np

def split_sequence(sequence, n_steps):
    X = []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        seq_x = sequence[i:end_ix]
        X.append(seq_x)

    return np.array(X)

def normalize(data):
    return (data - 132) / 235

def show_profits(episode=0):
    logs = env.logs[episode]

    actions = np.array(logs['actions'])
    profits = np.array(logs['profit'])
    indices = np.arange(0,len(actions))
    stocks  = df.values[WINDOW_SIZE-1:]

    logs = pd.DataFrame({
        'action': actions,
        'profit': profits, 
        'stock': stocks,
        'bought': actions == BUY,
        'sold': actions == SELL,
        'held': actions == HOLD}).reset_index()

    logs['action_cat'] = logs['action'].replace({0: 'Buy', 1: 'Sell', 2: 'Hold'})
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=logs['index'], 
                             y=logs['profit'],
                             mode='lines',
                             name='Profits'))
    
    fig.add_trace(go.Scatter(x=logs['index'], 
                         y=logs['stock'],
                         mode='lines',
                         name='Stock'))

    bought = logs[logs['bought']]
    fig.add_trace(go.Scatter(x=bought['index'], 
                             y=bought['profit'],
                             mode='markers',
                             name='Bought'))

    sold = logs[logs['sold']]
    fig.add_trace(go.Scatter(x=sold['index'], 
                             y=sold['profit'],
                             mode='markers',
                             name='Sold'))

    fig.show()
    
    fig = go.Figure(data=[go.Pie(labels=logs['action_cat'].value_counts().index,
                             values=logs['action_cat'].value_counts().values)])
    fig.show()