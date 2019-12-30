import numpy as np

EPSILON = 1e-6


def logsafe(X):
    threshold = -100
    X = np.log(X)
    # X = np.nan_to_num(X)
    X = (1 - np.isinf(X)) * np.nan_to_num(X) + np.isinf(X) * threshold
    return X

def log_expand_dims(X, axis=-1):
    X = logsafe(X)
    X = np.expand_dims(X, axis)
    return X

def normalize(X, axis=1):
    z = np.sum(X, axis=axis, keepdims=True)
    z = z + (z==0) * EPSILON

    X = X / z
    return X

def normalize_exp(X, axis=1):
    X = X - np.max(X)
    X = normalize(np.exp(X), axis)
    return X

def unique_states(sequences):
    states = set()

    for seq in sequences:
        for state in seq:
            if state not in states:
                states.update(state)
    return list(states)
