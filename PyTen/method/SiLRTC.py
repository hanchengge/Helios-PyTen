__author__ = 'Song'

import numpy as np
import math
from PyTen.tenclass import tensor,tenmat


def Pro2TraceNorm(Z, tau):
    m=Z.shape[0]
    n=Z.shape[1]
    if 2*m < n:
        [U,Sigma2,V]=np.linalg.svd(np.dot(Z,Z.T))
        S=np.sqrt(Sigma2)
        tol = np.max(Z.shape) * (2**int(math.log(max(S),2)))*2.2204*1E-16
        n = np.sum(S > max(tol, tau))
        mid = [max(S[i]-tau,0)*1.0/S[i] for i in range(n)]
        X = np.dot(np.dot(U[:, 0:n],np.dot(np.diag(mid),U[:, 0:n].T)),Z)
        return X, n, Sigma2
    if m > 2*n:
        [X, n, Sigma2] = Pro2TraceNorm(Z.T, tau)
        X = X.T
        return X, n, Sigma2
    [U,S,V] = np.linalg.svd(Z)
    Sigma2 = S**2
    n = sum(S > tau)
    X = np.dot(U[:, 0:n],np.dot(np.diag(S[0:n]-tau),V[0:n,:]))
    return X, n, Sigma2


def SiLRTC(X, Omega=None, alpha=None, gamma=None, maxIter=500, epsilon=1e-5,printitn=1):
    """ Simple Low Rank Tensor Completion (SiLRTC).
        Reference: "Tensor Completion for Estimating Missing Values. in Visual Data", PAMI, 2012."""

    T=X.data.copy()
    N = X.ndims
    dim = X.shape
    if printitn==0:
        printitn=maxIter
    if Omega==None:
        Omega=X.data*0+1

    if alpha==None:
        alpha=np.ones([N])
        alpha=alpha/sum(alpha)

    if gamma==None:
        gamma = 0.1*np.ones([N])

    normX = X.norm()
    # initialization
    X.data[Omega==0] = np.mean(X.data[Omega==1])
    errList = np.zeros([maxIter, 1])


    M = range(N)
    gammasum = sum(gamma)
    tau = alpha/ gamma

    for k in range(maxIter):
        if (k+1)%printitn==0:
            print 'SiLRTC: iterations = %d   difference=%f\n', k, errList[k-1]

        Xsum = 0
        for i in range(N):
            temp=tenmat.tenmat(X,i+1)
            [temp1,tempn,tempSigma2]=Pro2TraceNorm(temp.data, tau[i])
            temp.data=temp1
            M[i] = temp.totensor().data
            Xsum = Xsum + gamma[i] * M[i]

        Xlast = X.data.copy()
        Xlast = tensor.tensor(Xlast)

        X.data = Xsum / gammasum
        X.data[Omega==1] = T[Omega==1]
        diff=X.data-Xlast.data
        errList[k] = np.linalg.norm(diff) / normX
        if errList[k] < epsilon:
            errList = errList[0:k]
            break

    print 'SiLRTC ends: total iterations = %d   difference=%f\n\n', k, errList[k-1]
    return X, errList