from GPy.kern.src.kern import Kern
import numpy as np
from GPy.core.parameterization import Param
# from paramz.caching import Cache_this

# from cnn_kernel import kernel_matrix

class CustomMatrix(Kern):
    def __init__(self,input_dim,X,K,active_dims=None):
    # def __init__(self,input_dim,active_dims=None):
        super(CustomMatrix, self).__init__(input_dim, active_dims, 'custom_matrix')
        self.Kmatrix = K #Param('matrix', K)
        self.X = X
        # self.link_parameters(self.K)
    # @Cache_this(limit=3, ignore_args=())
    def K(self,X1,X2):
        # return kernel_matrix(X,X2)
        if X2 is None and np.all(X1 == self.X):
            return self.Kmatrix

        indices1 = np.concatenate([np.nonzero(np.prod(self.X == x,1))[0] for x in X1])

        if X2 is None:
            X2 = X1
            indices2 = indices1
        else:
            indices2 = np.concatenate([np.nonzero(np.prod(self.X == x,1))[0] for x in X2])

        if np.all(np.isin(X2,self.X)):
            if len(indices2) != X2.shape[0] or len(indices1) != X1.shape[0]:
                print(indices2, indices1)
                raise NotImplementedError("Some elements of X2 or X1 appear more than once in X")
            else:
                return self.Kmatrix[indices1[:, None], indices2]
        else:
            raise NotImplementedError("Some elements of X2 are not in X")
        #     return NotImplementedError
    # @Cache_this(limit=3, ignore_args=())
    def Kdiag(self,X):
        # K = kernel_matrix(X,X2)
        indices = np.concatenate([np.nonzero(np.prod(self.X == x,1))[0] for x in X])
        return np.diag(self.Kmatrix)[indices]
    def update_gradients_full(self, dL_dK, X, X2):
        pass
    def update_gradients_diag(self, dL_dK, X, X2):
        pass
