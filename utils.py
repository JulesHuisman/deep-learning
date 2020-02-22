import numpy as np

from keras.models import model_from_config, Sequential, Model

def split_sequence(sequence, n_steps):
    """
    Create timeseries from array
    """

    X = []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        seq_x = sequence[i:end_ix]
        X.append(seq_x)

    return np.array(X)

def clone_model(model):
    """
    Clone a keras model
    Adjusted from https://github.com/keras-rl/keras-rl/blob/216c3145f3dc4d17877be26ca2185ce7db462bad/rl/util.py#L8
    """
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }

    clone = model_from_config(config)
    clone.set_weights(model.get_weights())

    return clone