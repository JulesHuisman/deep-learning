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
