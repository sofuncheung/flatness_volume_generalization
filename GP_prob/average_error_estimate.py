import numpy as np
def average_error(K,sigma=1e-1):
    eigs = np.linalg.eigh(K)[0]
    m=K.shape[0]
    epsilon = 1
    for i in range(1000):
        epsilon = np.sum(eigs/(1+eigs*m/(sigma**2+epsilon)))
    return epsilon
