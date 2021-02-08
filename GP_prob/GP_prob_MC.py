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

def GP_prob(K,X,Y, method="importance_sampling"):
    # globals().update(FLAGS)
    if method=="importance_sampling":
        # link_fun = GPy.likelihoods.link_functions.Heaviside()
        # lik = GPy.likelihoods.Bernoulli(gp_link=link_fun)
        lik = GPy.likelihoods.Bernoulli()

        inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=True)
        # inference_method = GPy.inference.latent_function_inference.laplace.Laplace()

        model = GPy.core.GP(X=X,
                        Y=Y,
                        kernel=CustomMatrix(X.shape[1],X,K),
                        inference_method=inference_method,
                        likelihood=lik)

        mean, cov = model._raw_predict(X, full_cov=True)
        mean *= mean_mult
        mean = mean.flatten()
        cov *= cov_mult
        log_norm_ratio = np.sum(np.log((np.linalg.eigh(cov)[0] / np.linalg.eigh(K)[0])))/2
        covinv = np.linalg.inv(cov)
        Kinv = np.linalg.inv(K)
        tot = 0
        shift = m*np.log(2)*0.3

        #parallelizing tasks with mpi rank
        num_tasks = num_post_samples
        num_tasks_per_job = num_tasks//size
        tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
        if rank < num_tasks%size:
            tasks.append(size*num_tasks_per_job+rank)

        sample = np.random.multivariate_normal(mean, cov, len(tasks)).T

        import sys
        for i in range(len(tasks)):
            # print(i,len(tasks))
            # if (i%int(len(tasks)/100)) == (int(len(tasks)/100)-1):
            #     print(str(int(100*i/len(tasks)))+"%")
            #     # print(f)
            #     # print(Y.T)
            #     sys.stdout.flush()
            f = sample[:,i]
            PQratio = np.exp(shift-0.5*(np.matmul(f.T, np.matmul(Kinv, f)) - np.matmul((f-mean).T, np.matmul(covinv, (f-mean))) ) + log_norm_ratio)
            if np.prod((f.T>0) == Y.T):
                print(PQratio)
                sys.stdout.flush()
                tot += PQratio

        tots = comm.gather(tot,root=0)
        if rank == 0:
            tot = sum(tots)
            logPU = np.log(tot) - shift - np.log(num_post_samples)
            return logPU
        else:
            return None
    elif method == "HMC":
        import gpflow
        import custom_kernel_gpflow
        from custom_kernel_gpflow import CustomMatrix

        m = gpflow.models.GPMC(X.astype(np.float64), Y,
            kern=CustomMatrix(X.shape[1],X,K),
            # kern=gpflow.kernels.RBF(28*28),
            likelihood=gpflow.likelihoods.Bernoulli(),)

        m.compile()
        o = gpflow.train.AdamOptimizer(0.01)
        o.minimize(m, maxiter=1000) # start near MAP

        s = gpflow.train.HMC()

        samples = s.sample(m, 10, epsilon=1e-4, lmax=7, lmin=2, thin=1, logprobs=True)#, verbose=True)

        n = samples.iloc[0][0].shape[0]
        N = len(samples)
        def hasZeroLikelihood(f):
            return np.any(np.sign(f) != Y*2-1)

        #Using MCMC estimate from https://stats.stackexchange.com/a/210196/32021 (with high prob region being the intersection of ball of radius
        # sqrt(n) and the non-zero-likelihood quadrant.)
        tot=0
        for i, (f,logP) in samples.iterrows():
            if not hasZeroLikelihood(f) and np.linalg.norm(f) <= np.sqrt(n):
                tot += np.exp(-logP)

        # !pip install -U scipy
        import importlib; importlib.reload(scipy)
        import scipy
        from scipy import special
        logV = (n/2)*np.log(n)+(n/2)*np.log(np.pi)-np.log(special.gamma(n/2+1))-n*np.log(2)
        return logV - np.log(tot/N)

    elif method == "Metropolis_EPproposal":
        assert Y>=0 # not -1,1
        Kinv = np.linalg.inv(K)

        det = np.linalg.eigh(K)[0]
        n = len(X)
        normalization = (np.sqrt(np.power(2*np.pi,n) * det))
        lognorm = 0.5*(len(X)*np.log(2*np.pi)+np.sum(np.log(det)))

        def logPtilde(f):
            return -0.5*(np.matmul(f.T, np.matmul(Kinv, f))) - lognorm

        def logProposal(f2,f1):
            return -0.5*(np.matmul(f.T, np.matmul(np.eye(n), f)))/sigma**2 - lognorm

        def newProposal(f1):
            #return np.random.multivariate_normal(f1,sigma*np.eye(n))
            return np.random.multivariate_normal(f1,sigma*K)

        def hasZeroLikelihood(f):
            return np.any(np.sign(f) != Y*2-1)

        def alpha(logPf2,logPf1):
            if hasZeroLikelihood(f2):
                return 0
            else:

                logRatio = logPf2-logPf1
                return min(1,np.exp(logRatio))

        sigma=0.05
        import scipy
        # V=np.power(n,n/2)*np.power(np.pi,n/2)/scipy.special.gamma(n/2+1)/np.power(2,n)
        logV = (n/2)*np.log(n)+(n/2)*np.log(np.pi)-np.log(scipy.special.gamma(n/2+1))-n*np.log(2)
        f1 = np.squeeze(Y*2-1)
        tot = 0
        N = 10000
        accepted = 0
        for i in range(N):
            f2 = newProposal(f1)
            logPf2 = logPtilde(f2)
            logPf = logPf1 = logPtilde(f1)
            if np.random.rand() <= alpha(logPf2,logPf1):
                f1 = f2
                logPf = logPf2
                accepted += 1
            if np.linalg.norm(f1) <= np.sqrt(n):
                tot += np.exp(-logPf)
                # print(i)
                # print(i,",".join([str(x) for x in f1]))


        return logV - np.log(tot/N)
    else:
        raise NotImplementedError()
