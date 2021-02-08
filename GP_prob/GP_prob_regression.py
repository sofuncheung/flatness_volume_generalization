import GPy
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
GP_prob_folder = os.path.join(ROOT_DIR, 'GP_prob')
sys.path.append(GP_prob_folder)
import numpy as np
from numpy.linalg import inv
from numpy import matmul

# import neural_tangents as nt
# from neural_tangents import stax

def GP_prob(K,X,Y,sigma_noise=1.0, posterior="bayes"):
    n = X.shape[0]
    if posterior=="bayes":
        alpha = matmul(inv(np.eye(n)*(sigma_noise**2)+K),Y)
        cov = inv(inv(K)+np.eye(n)/(sigma_noise**2))
    elif posterior=="ntk":
        pass
        # mean =
        # cov =
        # alpha = np.linalg.solve(K,mean)
    covi = inv(cov)
    coviK = matmul(covi,K)
    KL = 0.5*np.log(np.linalg.det(coviK)) + 0.5*np.trace(inv(coviK)) + 0.5*matmul(alpha.T,matmul(K,alpha)) - n/2
    return -KL[0,0]

'''PLAYGROUND'''


    # import scipy
    # t = 0.005
    # t = 1.0
    # import numpy as np
    # decay_matrix = np.eye(n)-scipy.linalg.expm(-t*theta)
    # decay_matrix.shape
    # theta.shape
    # K.shape
    # Sigma = K + np.matmul(decay_matrix, np.matmul(K, np.matmul(np.linalg.inv(theta), np.matmul(decay_matrix, theta))) - 2*K)
    #
    # Y = y_train.reshape(n)
    # alpha = np.matmul(np.linalg.inv(K), np.matmul(decay_matrix,Y))
    #
    # # eigs_sigma = np.maximum(1e-3,np.linalg.eigh(Sigma)[0])
    # # eigs_sigma = np.linalg.eigh(Sigma)[0]
    # eigs_sigma = np.linalg.eigvals(Sigma)
    # # eigs_K = np.maximum(1e-3,np.linalg.eigh(K)[0])
    # # eigs_K = np.linalg.eigh(K)[0]
    # eigs_K = np.linalg.eigvals(K)
    # KL = 0.5*(np.sum(np.log(1/eigs_sigma) + np.log(eigs_K)) + np.sum(eigs_sigma/eigs_K) + np.matmul(alpha.T,np.matmul(K,alpha)) - n)

    #lik = GPy.likelihoods.Bernoulli()
    #m = GPy.models.GPClassification(X=X,
    #                Y=Y,
    #                kernel=CustomMatrix(X.shape[1],X,K),
    #                # inference_method=GPy.inference.latent_function_inference.PEP(alpha = 1), #only for regression apparently
    #                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
    #                likelihood=lik)
# lik = GPy.likelihoods.Bernoulli()
# m = GPy.core.GP(X=X,
#                 Y=Y,
#                 kernel=CustomMatrix(X.shape[1],X,K),
#                 #inference_method=GPy.inference.latent_function_inference.PEP(alpha = 0.5),
#                 # inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=True,epsilon=1e-5),
#                 inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=True),
#                 likelihood=lik)
# # m.likelihood = lik
# #m.inference_method = GPy.inference.latent_function_inference.PEP(alpha = 0.5)
# m.log_likelihood()

# import custom_kernel_matrix
# import imp
# import custom_kernel_matrix.custom_kernel_matrix
# imp.reload(custom_kernel_matrix.custom_kernel_matrix)
# import numpy as np
# lik = GPy.likelihoods.Bernoulli()
# X = np.random.rand(200,1)
# k = GPy.kern.RBF(1, variance=7., lengthscale=0.2)
# f = np.random.multivariate_normal(np.zeros(200), k.K(X))
# p = lik.gp_link.transf(f) # squash the latent function
# Y = lik.samples(f).reshape(-1,1)
# m = GPy.models.GPClassification(X=X,
#                 Y=Y,
#                 kernel=CustomMatrix(X.shape[1],X,np.eye(X.shape[0])),
#                 # kernel=k,
#                 # inference_method=GPy.inference.latent_function_inference.PEP(alpha = 1),
#                 inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
#                 likelihood=lik)
# # m = GPy.models.GPClassification(X=X,
# #                 Y=Y,
# #                 # kernel=CustomMatrix(X.shape[1],K),
# #                 kernel=k,
# #                 inference_method=GPy.inference.latent_function_inference.PEP(alpha=1),
# #                 likelihood=lik)
# m.log_likelihood()
#
# X = np.random.rand(200,3)
# X2 = X[np.random.choice(range(0,200),replace=False, size=100)]
# np.where()
#
# X2
# np.where(np.prod(np.isin(X,X2),-1))
#
# indices = np.prod(np.isin(X,X2),-1).nonzero()[0]
#
#
# np.prod([False,False,False])
#
# X in X2
#
# K=np.random.randn(200,200)
#
# K[indices[:, None], indices].shape
#
# np.all(np.isin(X2,X))
