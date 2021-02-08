import numpy as np
import scipy.linalg as la

def pi(y,f):
    return 1/(1+np.exp(-y*f))

#Laplace approximation
def GP_prob(K,ys,tolerance=0.001):
    N = len(ys)
    #ys = np.array([int(x)*2-1 for x in fun])

    #fs_new = np.zeros(2**n)
    # fs_new = np.zeros(len(ys))
    fs_new = ys

    fs = fs_new
    #     print(fs)
    W = np.diag([pi(1,fs[i])*(1-pi(1,fs[i])) for i,y in enumerate(ys)])
    B = np.eye(N) + np.matmul(W**0.5,np.matmul(K,W**0.5))
    L = np.linalg.cholesky(B)
    b = np.dot(W,fs) + np.array([(y+1)/2 - pi(1,fs[i]) for i,y in enumerate(ys)])
    a = b - np.matmul(W**0.5,la.solve_triangular(L.T,la.solve_triangular(L,np.dot(W**0.5,np.dot(K,b)),lower=True)))
    fs_new = np.dot(K,a)
    objective_new = -0.5*np.dot(a,fs_new)+ np.sum([np.log(pi(y,fs_new[i])) for i,y in enumerate(ys)])

    objective_old = objective_new
    fs = fs_new
    #     print(fs)
    W = np.diag([pi(1,fs[i])*(1-pi(1,fs[i])) for i,y in enumerate(ys)])
    B = np.eye(N) + np.matmul(W**0.5,np.matmul(K,W**0.5))
    L = np.linalg.cholesky(B)
    b = np.dot(W,fs) + np.array([(y+1)/2 - pi(1,fs[i]) for i,y in enumerate(ys)])
    a = b - np.matmul(W**0.5,la.solve_triangular(L.T,la.solve_triangular(L,np.dot(W**0.5,np.dot(K,b)),lower=True)))
    fs_new = np.dot(K,a)
    objective_new = -0.5*np.dot(a,fs_new)+ np.sum([np.log(pi(y,fs_new[i])) for i,y in enumerate(ys)])

    # while np.linalg.norm(fs-fs_new) > 0.01:
    while abs(objective_old - objective_new) > tolerance:
        print(objective_new)
        objective_old = objective_new
        fs = fs_new
    #         print(fs)
        W = np.diag([pi(1,fs[i])*(1-pi(1,fs[i])) for i,y in enumerate(ys)])
        B = np.eye(N) + np.matmul(W**0.5,np.matmul(K,W**0.5))
        L = np.linalg.cholesky(B)
        b = np.dot(W,fs) + np.array([(y+1)/2 - pi(1,fs[i]) for i,y in enumerate(ys)])
        a = b - np.matmul(W**0.5,la.solve_triangular(L.T,la.solve_triangular(L,np.dot(W**0.5,np.dot(K,b)),lower=True)))
        fs_new = np.dot(K,a)
        objective_new = -0.5*np.dot(a,fs_new)+ np.sum([np.log(pi(y,fs_new[i])) for i,y in enumerate(ys)])

    logProb = objective_new - np.sum(np.log(np.diag(L)))
    return logProb

# kernel = np.load(open("kernel_20k_mnist_fc.p","rb"))
#
# train_label = np.load(open("train_label_20k_mnist.p","rb"))
#
# ys = [int((np.argmax(labels)>5))*2.0-1 for labels in train_label]
#
# logPU = GP_prob(kernel,ys,0.001)
#
# m=20000
# delta = 2**-10
# (-logPU+2*np.log(m)+1-np.log(delta))/m
