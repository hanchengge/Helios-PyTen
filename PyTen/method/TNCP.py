#!/usr/bin/env python
__author__ = 'Qingquan Song'
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
from scipy.sparse import csgraph
from PyTen.tenclass import tensor
from PyTen.tenclass import tenmat
from PyTen.tools import tendiag

class TNCP(object):

# This routine solves the nuclear-norm regularized CP tensor completion problem
# via Alternation Direction Method of Multipliers (ADMM), which has been
# presented in our papers:
# 1. Yuanyuan Liu, Fanhua Shang, Hong Cheng, James Cheng, Hanghang Tong:
# Factor Matrix Trace Norm Minimization for Low-Rank Tensor Completion,
# SDM, pp. 866-874, 2014.
#
# 2. Yuanyuan Liu, Fanhua Shang, L. C. Jiao, James Cheng, Hong Cheng:
# Trace Norm Regularized CANDECOMP/PARAFAC Decomposition with Missing Data,
# accepted by IEEE Transactions on Cybernetics, 2015.

    def __init__(self, obser, omega=None,  rank=20, tol=1e-5, maxIter=500, alpha=None, lmbda=None, eta=1e-4, rho=1.05):
        if not obser:
            raise ValueError("TNCP: observed tensor cannot be empty!")
        elif type(obser) != tensor.tensor and type(obser) != np.ndarray:
            raise ValueError("TNCP: cannot recognize the format of observed tensor!")
        elif type(obser) == np.ndarray:
            self.T = tensor.tensor(obser)
        else:
            self.T = obser

        if omega==None:
            self.omega=self.T.data*0+1
        if type(omega) != tensor.tensor and type(omega) != np.ndarray:
            raise ValueError("TNCP: cannot recognize the format of indicator tensor!")
        elif type(omega) == np.ndarray:
            self.Omega = tensor.tensor(omega)
        else:
            self.Omega = omega

        if not self.Omega:
            raise ValueError("TNCP: indicator tensor cannot be empty!")

        self.ndims = self.T.ndims
        self.shape = self.T.shape


        if alpha is None:
            self.alpha = np.ones(self.ndims)
            self.alpha = self.alpha / sum(self.alpha)
        else:
            self.alpha = alpha

        self.rank = rank

        if lmbda is None:
            self.lmbda = 1/np.sqrt(max(self.shape))
        else:
            self.lmbda = lmbda

        self.maxIter = maxIter
        self.tol = tol
        self.eta = eta
        self.rho = rho
        self.errList = []
        self.X = None
        self.X_pre = None
        self.U = None
        self.Y = None
        self.Z = None
        self.II = None
        self.normT = np.linalg.norm(self.T.data)

    def initializeLatentMatrices(self):
        self.U = [np.random.rand(self.shape[i], self.rank) for i in range(self.ndims)]
        self.Y = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
        self.Z = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
        self.II = tendiag.tendiag(np.ones(self.rank), [self.rank for i in range(self.ndims)])
        self.X = self.T.data + (1-self.Omega.data)*(self.T.norm()/self.T.size())
        self.X = tensor.tensor(self.X)
        self.X_pre = self.X.copy()

    def run(self):
        self.errList = []

        self.initializeLatentMatrices()

        for k in range(self.maxIter):
            if k%1 == 1:
                print 'TNCP: iterations={0}, difference={1}'.format(k, self.errList[-1])

            # update step eta
            self.eta *= self.rho

            # update Z
            for i in range(self.ndims):
                temp_1 = self.U[i] - self.Y[i]/self.eta
                U, S, V = np.linalg.svd(temp_1)
                for j in range(S.size):
                    S[j] = max(S[j],self.alpha[i]/self.eta)
                [m,n]=temp_1.shape
                if m>n:
                    S=np.dot(np.eye(m,n),np.diag(S))
                else:
                    S=np.dot(np.diag(S),np.eye(m,n))
                self.Z[i] = np.dot(np.dot(U,S),V)
            temp_1, U, S, V = None, None, None, None

            # update U
            for i in range(self.ndims):
                # calculate intermedian tensor and its mode-n unfolding
                midT = self.II.copy()
                # calculate Kronecker product of U(1), ..., U(i-1),U(i+1), ...,U(n)
                for j in range(self.ndims):
                    if j == i:
                        continue
                    midT=midT.ttm(self.U[j], j+1)
                unfoldD_temp = tenmat.tenmat(midT, i+1)

                temp_Z = self.eta*self.Z[i] + self.Y[i]
                temp_B = np.dot(unfoldD_temp.data,unfoldD_temp.data.T)
                temp_B += self.eta*np.identity(self.rank)
                temp_B += 0.00001*np.identity(self.rank)
                temp_C = tenmat.tenmat(self.X, i+1)
                temp_D = np.dot(temp_C.data,unfoldD_temp.data.T)
                self.U[i] = np.dot((temp_D + temp_Z), np.linalg.inv(temp_B))

            midT, unfoldD_temp, temp_Z, temp_B, temp_C, temp_D = None, None, None, None, None, None

            # update X
            midT = self.II.copy()
            for i in range(self.ndims):
                midT=midT.ttm(self.U[i], i+1)
            self.X = midT.copy()
            self.X.data = self.T.data * self.Omega.data + self.X.data * (1 - self.Omega.data)

            midT = None

            # update Lagrange multiper
            for i in range(self.ndims):
                self.Y[i] += self.eta * (self.Z[i] - self.U[i])

            # checking the stop criteria
            error = np.linalg.norm(self.X_pre.data - self.X.data)/self.normT
            self.X_pre = self.X.copy()
            self.errList.append(error)

            if error < self.tol:
                break