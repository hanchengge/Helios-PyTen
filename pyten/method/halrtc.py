__author__ = 'Song'


import numpy as np
import math
import pyten.tenclass


def pro_to_trace_norm(Z, tau):
    m=Z.shape[0]
    n=Z.shape[1]
    if 2*m < n:
        [U,Sigma2,V]=np.linalg.svd(np.dot(Z,Z.T))
        S=np.sqrt(Sigma2)
        tol = np.max(Z.shape) * (2**int(math.log(max(S),2)))*2.2204*1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i]-tau,0)*1.0/S[i] for i in range(k)]
        X = np.dot(np.dot(U[:, 0:k],np.dot(np.diag(mid),U[:, 0:k].T)),Z)
        return X, k, Sigma2
    if m > 2*n:
        Z=Z.T
        [U,Sigma2,V]=np.linalg.svd(np.dot(Z,Z.T))
        S=np.sqrt(Sigma2)
        tol = np.max(Z.shape) * (2**int(math.log(max(S),2)))*2.2204*1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i]-tau,0)*1.0/S[i] for i in range(k)]
        X = np.dot(np.dot(U[:, 0:k],np.dot(np.diag(mid),U[:, 0:k].T)),Z)
        return X.T, k, Sigma2

    [U,S,V] = np.linalg.svd(Z)
    Sigma2 = S**2
    k = sum(S > tau)
    X = np.dot(U[:, 0:k],np.dot(np.diag(S[0:k]-tau),V[0:k,:]))
    return X, n, Sigma2



def halrtc(X, Omega=None, alpha=None, beta=None, maxIter=100, epsilon=1e-5,printitn=100):
    """ High Accuracy Low Rank Tensor Completion (HaLRTC).
        Reference: "Tensor Completion for Estimating Missing Values. in Visual Data", PAMI, 2012."""

    T=X.data.copy()
    N = X.ndims
    dim = X.shape
    normX = X.norm()

    if printitn==0:
        printitn=maxIter
    if Omega is None:
        Omega=X.data*0+1

    if alpha is None:
        alpha=np.ones([N])
        alpha=alpha/sum(alpha)

    if beta is None:
        beta=1e-6


    # Initialization
    X.data[Omega==0] = np.mean(X.data[Omega==1])
    errList = np.zeros([maxIter, 1])

    Y = range(N)
    M = range(N)

    for i in range(N):
        Y[i] = X.data
        M[i] = np.zeros(dim)


    Msum = np.zeros(dim)
    Ysum = np.zeros(dim)

    for k in range(maxIter):
        if (k+1) % printitn == 0 and k != 0 and printitn != maxIter:
            print 'HaLRTC: iterations = {0}  difference = {0}\n'.format(k, errList[k-1])

        beta = beta * 1.05

        # Update Y
        Msum = 0*Msum
        Ysum = 0*Ysum
        for i in range(N):
            A=pyten.tenclass.Tensor(X.data-M[i]/beta)
            temp=pyten.tenclass.Tenmat(A,i+1)
            [temp1,tempn,tempSigma2]=pro_to_trace_norm(temp.data, alpha[i]/beta)
            temp.data=temp1
            Y[i] = temp.totensor().data
            Msum = Msum + M[i]
            Ysum = Ysum + Y[i]

        # update X
        Xlast = X.data.copy()
        Xlast = pyten.tenclass.Tensor(Xlast)
        X.data = (Msum + beta*Ysum) / (N*beta)
        X.data = T * Omega + X.data * (1-Omega)

        # Update M
        for i in range(N):
            M[i] = M[i] + beta*(Y[i] - X.data)


        # Compute the error
        diff=X.data-Xlast.data
        errList[k] = np.linalg.norm(diff) / normX
        if errList[k] < epsilon:
            errList = errList[0:(k+1)]
            break

    print 'HaLRTC ends: total iterations = {0}   difference = {1}\n\n'.format(k+1, errList[k])
    return X
