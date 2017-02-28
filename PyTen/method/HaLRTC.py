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



def HaLRTC(X, Omega=None, alpha=None, beta=None, maxIter=500, epsilon=1e-5,printitn=1):
    """ High Accuracy Low Rank Tensor Completion (HaLRTC).
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

    if beta==None:
        beta=1e-6

    normX = X.norm()
    # initialization
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
        if (k+1)%printitn==0:
            print 'HaLRTC: iterations = %d   difference=%f\n', k, errList[k-1]

        beta = beta * 1.05;

        # update Y
        Msum = 0*Msum
        Ysum = 0*Ysum
        for i in range(N):
            A=tensor.tensor(X.data-M[i]/beta)
            temp=tenmat.tenmat(A,i+1)
            [temp1,tempn,tempSigma2]=Pro2TraceNorm(temp.data, alpha[i]/beta)
            temp.data=temp1
            Y[i] = temp.totensor().data
            Msum = Msum + M[i]
            Ysum = Ysum + Y[i]

        # update X
        Xlast = X.data.copy()
        Xlast = tensor.tensor(Xlast)
        X.data = (Msum + beta*Ysum) / (N*beta)
        X.data[Omega==1] = T[Omega==1]

        # update M
        for i in range(N):
            M[i] = M[i] + beta*(Y[i] - X.data)


        # compute the error
        diff=X.data-Xlast.data
        errList[k] = np.linalg.norm(diff) / normX
        if errList[k] < epsilon:
            errList = errList[0:k]
            break

    print 'HaLRTC ends: total iterations = %d   difference=%f\n\n', k, errList[k-1]
    return X, errList