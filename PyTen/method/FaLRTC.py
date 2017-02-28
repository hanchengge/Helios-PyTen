__author__ = 'Song'


import numpy as np
import math
from PyTen.tenclass import tensor,tenmat

def SingularValue(A):
    m=A.shape[0]
    n=A.shape[1]
    if 2*m < n:
        AAT = np.dot(A,A.T)
        [U, S, V] = np.linalg.svd(AAT)
        S = np.sqrt(S)
        return S
    if m > 2*n:
        AAT = np.dot(A.T,A)
        [U, S, V] = np.linalg.svd(AAT)
        S = np.sqrt(S)
        return S
    [U,S,V] = np.linalg.svd(A)
    return S

def Truncate(Z, tau):
    #Z is a tenmat
    m=Z.shape[0]
    n=Z.shape[1]
    if 2*m<n:
        [U,Sigma2,V]=np.linalg.svd(np.dot(Z,Z.T))
        S=np.sqrt(Sigma2)
        tol = np.max(Z.shape) * (2**int(math.log(max(S),2)))*2.2204*1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i]-tau,0)*1.0/S[i] for i in range(k)]
        X = np.dot((np.eye(m)-np.dot(U[:, 0:k],np.dot(np.diag(mid),U[:, 0:k].T))),Z)
        return X,Sigma2

    if 2*m>n:
        Z=Z.T
        [U,Sigma2,V]=np.linalg.svd(np.dot(Z,Z.T))
        S=np.sqrt(Sigma2)
        tol = np.max(Z.shape) * (2**int(math.log(max(S),2)))*2.2204*1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i]-tau,0)*1.0/S[i] for i in range(k)]
        X = np.dot((np.eye(n)-np.dot(U[:, 0:k],np.dot(np.diag(mid),U[:, 0:k].T))),Z)
        return X.T,Sigma2

    [U,sigma,V] = np.linalg.svd(Z)
    Sigma2 = sigma**2
    n = np.sum(sigma > tau)
    X = Z - np.dot(U[:, 0:n],np.dot(np.diag(sigma[0:n]-tau),V[0:n,:]))
    return X, Sigma2

#X=X1; Omega=Omega1; alpha=None; mu=None; L=1e-5; C=0.6; maxIter=100; epsilon=1e-5;printitn=1

def FaLRTC(X, Omega=None, alpha=None, mu=None, L=1e-5, C=0.6, maxIter=500, epsilon=1e-5,printitn=1):
    """ Fast Low Rank Tensor Completion (FaLRTC).
        Reference: "Tensor Completion for Estimating Missing Values. in Visual Data", PAMI, 2012."""
    N = X.ndims
    dim = X.shape
    if printitn==0:
        printitn=maxIter
    if Omega==None:
        Omega=X.data*0+1

    if alpha==None:
        alpha=np.ones([N])
        alpha=alpha/sum(alpha)

    if mu==None:
        mu = 5.0 * alpha / np.sqrt(dim)

    normX = X.norm()
    # initialization
    X.data[Omega==0] = np.mean(X.data[Omega==1])

    Y=X.data.copy()
    Y=tensor.tensor(Y)
    Z=X.data.copy()
    Z=tensor.tensor(Z)
    B = 0

    Gx = np.zeros(dim)
    errList = np.zeros([maxIter, 1])

    Lmax = 10*np.sum(1.0/mu)

    tmp = np.zeros([N])
    for i in range(N):
        tempX=tenmat.tenmat(X,i+1)
        [U,sigma,V] = np.linalg.svd(tempX.data)
        #sigma=SingularValue(tempX.data)
        tmp[i] = np.max(sigma) * alpha[i] * 0.3

    P = 1.15
    flatNum = 15
    slope = (tmp - mu) / (1-(maxIter-flatNum)**(-P))
    offset = (mu*(maxIter-flatNum)**P - tmp) / ((maxIter-flatNum)**P-1)

    mu0 = mu*1.0
    for k in range(maxIter):
        if (k+1)%printitn==0:
             print 'FaLRTC: iterations = %d   difference=%f\n', k, errList[k-1]


        # update mu
        t=slope*1.0 /(k+1)**P+offset
        mu = [max( t[j],mu0[j])*1.0 for j in range(N)]

        a2m = alpha**2/ mu
        ma = mu/alpha

        Ylast = Y.data.copy()
        Ylast = tensor.tensor(Ylast)
        while True:
            b = (1+np.sqrt(1+4*L*B))*1.0 / (2.0*L)
            X.data = b*1.0/(B+b) * Z.data + B*1.0/(B+b) * Ylast.data

            # compute f'(x) namely "Gx" and f(x) namely "fx"
            Gx = Gx * 0
            fx = 0
            for i in range(N):
                temp=tenmat.tenmat(X,i+1)
                [tempX, sigma2] = Truncate(temp.data, ma[i])
                temp.data=tempX
                temp = temp.totensor()
                Gx = Gx + a2m[i] * temp.data
                #print sum(sigma2)
                #print (np.max(np.sqrt(sigma2)-ma[i], 0))
                #print (np.max(np.sqrt(sigma2)-ma[i], 0))**2
                #print np.sum( (np.max(np.sqrt(sigma2)-ma[i], 0))**2 )
                #print (sum(sigma2) - np.sum( (np.max(np.sqrt(sigma2)-ma[i], 0))**2 ) )
                #print a2m[i]*(sum(sigma2) - np.sum( (np.max(np.sqrt(sigma2)-ma[i], 0))**2 ) )
                fx = fx + a2m[i]*(sum(sigma2) - sum([(max(np.sqrt(a)-ma[i],0))**2 for a in sigma2]) )
            Gx[Omega==1] = 0

            # compute f(Ytest) namely fy
            Y.data = X.data - Gx / L
            fy = 0
            for i in range(N):
                tempY=tenmat.tenmat(Y,i+1)
                [U,sigma,V] = np.linalg.svd(tempY.data)
                #sigma=SingularValue(tempY.data)
                fy = fy + a2m[i]*(sum(sigma**2) - sum( ([ (max(q-ma[i],0))**2 for q in sigma]) ) )

            # test if L(fx-fy) > \|Gx\|^2
            #print fx,fy
            #print np.sum(Gx[:]**2)
            #print (fx - fy)*L
            if (fx - fy)*L < np.sum(Gx[:]**2):
                if L > Lmax:
                    print 'FaLRTC: iterations = %d   difference=%f\n Exceed the Maximum Lipschitiz Constan\n\n', k, errList[k-1]
                    errList = errList[0:k]
                    return Y,errList
                L = L/C
            else:
                 break

        # Check Convergence
        diff=Y.data-Ylast.data
        errList[k] = np.linalg.norm(diff) / normX
        if errList[k] < epsilon:
            break

        # update Z, Y, and B
        Z.data = Z.data - b*Gx
        B = B+b

    errList = errList[0:k]
    print 'FaLRTC ends: total iterations = %d   difference=%f\n\n', k, errList[k-1]
    return Y, errList