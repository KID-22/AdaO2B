import numpy as np
from scipy.special import expit

def get_batch_feedback(state, w):
    # state: B, d
    # w: d, 1
    prop = expit(np.dot(state, w)) # B, 1
    feedback = [np.random.choice([0, 1], p=[1 - prop[i][0], prop[i][0]]) for i in range(state.shape[0])]
    return np.array(feedback)