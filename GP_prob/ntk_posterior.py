import GPy
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
GP_prob_folder = os.path.join(ROOT_DIR, 'GP_prob')
sys.path.append(GP_prob_folder)
import numpy as np
from numpy.linalg import inv
from numpy import matmul
import scipy

# import neural_tangents as nt
# from neural_tangents import stax

def matmul_reduce(Ms):
    r = matmul(Ms[0], Ms[1])
    for i in range(2,len(Ms)):
        r = matmul(r, Ms[i])
    return r

#theta is ntk
#K is NNGP
def NTK_posterior(K_train,K_test,K_train_test,theta_train,theta_test,theta_train_test,X,Y,t=1.0):
    n = K_train.shape[0]
    # t = 1.0
    theta = theta_train
    theta_test_train = theta_train_test.T
    theta_inv = np.linalg.inv(theta)

    decay_matrix = np.eye(n)-scipy.linalg.expm(-t*theta)
    theta_decay_matrix = matmul_reduce([theta_test_train,theta_inv,decay_matrix])
    temp_var = matmul(theta_decay_matrix,K_train_test)
    Sigma = K_test + matmul_reduce([theta_decay_matrix,K_train,decay_matrix,theta_inv,theta_train_test]) - (temp_var + temp_var.T)

    mu = matmul(theta_decay_matrix,Y)
    return mu[:,0],Sigma
