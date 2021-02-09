# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

'''
Allow customized sigmaw and sigmab
'''

def kern(X1,X2,number_layers,sigmaw,sigmab, multiplier=1.0):
    '''
    Arguments:
        X1, X2:
            Type: tf.tensor
            shape: (number of examples, number of features)
        number_layers:
            Type: int
            The number of HIDDEN layers.
        sigmaw:
            Type: float
            1: When using Glorot initialization for each layer's weights
            np.sqrt(2): When using He_normal
        sigmab:
            Type: np.array
            length: number_layers+1
            The stddev of biases of each layer (from first hidden layer
            to output layer)

    '''
    N = X1.get_shape().as_list()[0]
    input_dim = X1.get_shape().as_list()[1]
    sigmaw *= multiplier
    if X2 is None:
        sigmab *= multiplier
        K = sigmab[0]**2 + sigmaw**2 * tf.matmul(X1,tf.transpose(X1))/input_dim
        for l in range(number_layers):
            K_diag = tf.linalg.diag_part(K)
            K_diag = tf.expand_dims(K_diag,-1)
            K1 = tf.tile(K_diag,[1,N])
            K2 = tf.tile(tf.transpose(K_diag),[N,1])

            K12 = K1 * K2
            costheta = K / tf.sqrt(K12)
            theta = tf.acos(costheta)
            sigmab *= multiplier
            K = sigmab[l+1]**2 + (sigmaw**2/(2*np.pi))*tf.sqrt(K12)*(tf.sin(theta)+(np.pi-theta)*costheta)

        # K_symm=K
        return K
    else:
        sigmab *= multiplier
        K = sigmab[0]**2 + sigmaw**2 * tf.matmul(X1,tf.transpose(X2))/input_dim
        K1_diag = sigmab[0]**2 + sigmaw**2 * tf.reduce_sum(X1*X1,axis=1, keepdims=True)/input_dim
        K2_diag = sigmab[0]**2 + sigmaw**2 * tf.reduce_sum(X2*X2,axis=1, keepdims=True)/input_dim
        for l in range(number_layers):
            # K1_diag = tf.expand_dims(K1_diag,-1)
            K1 = tf.tile(K1_diag,[1,N])
            # K2_diag = tf.expand_dims(K2_diag,-1)
            K2 = tf.tile(tf.transpose(K2_diag),[N,1])

            K12 = K1 * K2
            costheta = K / tf.sqrt(K12)
            theta = tf.acos(costheta)
            sigmab *= multiplier
            K = sigmab[l+1]**2 + (sigmaw**2/(2*np.pi))*tf.sqrt(K12)*(tf.sin(theta)+(np.pi-theta)*costheta)

            K1_diag = sigmab[l+1]**2 + (sigmaw**2/2)*K1_diag
            K2_diag = sigmab[l+1]**2 + (sigmaw**2/2)*K2_diag

        # K_cross=K
        return K


def kernel_matrix(X,number_layers,sigmaw,sigmab,n_gpus = 1, multiplier=1.0):
    m = X.shape[0]
    n_max = 10000
    n_max = min(n_max,m)
    for ii in range(n_max,0,-1):
        if m%ii==0:
            n_max = ii
            break
    slices = list((slice(j, j+n_max), slice(i, i+n_max))
        for j in range(0, m, n_max)
        for i in range(j, m, n_max))

    # X = train_images
    with tf.compat.v1.Session() as sess:
        K_ops = []
        if n_gpus>0:
            for i in range(n_gpus):
                with tf.device("gpu:{}".format(i)):
                    X1 = tf.compat.v1.placeholder(np.float64, [n_max, X.shape[1]], "X1")
                    X2 = tf.compat.v1.placeholder(np.float64, X1.shape, "X2")
                    K_cross = kern(X1, X2,number_layers,sigmaw,sigmab, multiplier)
                    K_symm = kern(X1, None,number_layers,sigmaw,sigmab, multiplier)
                    K_ops.append((X1, X2, K_cross, K_symm))
        else:
            with tf.device("cpu:{}".format(0)):
                X1 = tf.compat.v1.placeholder(np.float64, [n_max, X.shape[1]], "X1")
                X2 = tf.compat.v1.placeholder(np.float64, X1.shape, "X2")
                K_cross = kern(X1, X2,number_layers,sigmaw,sigmab, multiplier)
                K_symm = kern(X1, None,number_layers,sigmaw,sigmab, multiplier)
                K_ops.append((X1, X2, K_cross, K_symm))

        out = np.zeros((m, m), dtype=np.float64)
        # for j in tqdm.trange(0, len(slices), n_gpus):
        if n_gpus>0:
            n_devices = n_gpus
        else:
            n_devices = 1
        for j in range(0, len(slices), n_devices):
            feed_dict = {}
            ops = []
            for (X1, X2, K_cross, K_symm), (j_s, i_s) in (
                    zip(K_ops, slices[j:j+n_devices])):
                print((j_s, i_s))
                if j_s == i_s:
                    feed_dict[X1] = X[j_s]
                    ops.append(K_symm)
                else:
                    feed_dict[X1] = X[j_s]
                    feed_dict[X2] = X[i_s]
                    ops.append(K_cross)
            results = sess.run(ops, feed_dict=feed_dict)
            for r, (j_s, i_s) in zip(results, slices[j:j+n_devices]):
                out[j_s, i_s] = r
                if j_s != i_s:
                    out[i_s, j_s] = r.T
    return out

# np.save(open("kernel_20k_mnist_fc.p","wb"),out)
#
# np.save(open("train_label_20k_mnist.p","wb"),train_label)

# tf.reset_default_graph()
T1, T2, T3, T4 = 784, 40, 40, 1
sigmab = np.array([np.sqrt(2/T2), np.sqrt(2/T3), np.sqrt(2/T4)])

