import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import GPy
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
GP_prob_folder = os.path.join(ROOT_DIR, 'GP_prob')
sys.path.append(GP_prob_folder)
from custom_kernel_matrix.custom_kernel_matrix import CustomMatrix
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import gpflow
import custom_kernel_gpflow
from custom_kernel_gpflow import CustomMatrix

def GP_prob(K,X,Y):
    m = gpflow.models.GPMC(X.astype(np.float64), Y,
        kern=CustomMatrix(X.shape[1],X,K),
        # kern=gpflow.kernels.RBF(28*28),
        likelihood=gpflow.likelihoods.Bernoulli(),)
    m.compile()
    # print(m)
    return m.compute_log_likelihood()
