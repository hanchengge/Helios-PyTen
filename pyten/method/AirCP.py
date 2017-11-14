#!/usr/bin/env python
__author__ = "Hancheng Ge"
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
from scipy.sparse import csgraph
import pyten.tenclass
import pyten.tools


class AirCP(object):
    # This routine solves the auxiliary information regularized CP Tensor
    # completion via Alternation Direction Method of Multipliers (ADMM), which 
    # has been presented in our paper:

    # 1. Hancheng Ge, James Caverlee, Nan Zhang, Anna Squicciarini:
    # Uncovering the Spatio-Temporal Dynamics of Memes in the Presence of 
    # Incomplete Information, CIKM, 2016.

    def __init__(self, obser, omega=None, rank=20, tol=1e-5, maxIter=500, simMats=None, alpha=None, lmbda=None, eta=1e-4, rho=1.05, printitn=500):
        if not obser:
            raise ValueError("AirCP: observed Tensor cannot be empty!")
        elif type(obser) != pyten.tenclass.Tensor and type(obser) != np.ndarray:
            raise ValueError("AirCP: cannot recognize the format of observed Tensor!")
        elif type(obser) == np.ndarray:
            self.T = pyten.tenclass.Tensor(obser)
        else:
            self.T = obser

        if omega is None:
            self.omega = self.T.data*0+1

        if type(omega) != pyten.tenclass.Tensor and type(omega) != np.ndarray:
            raise ValueError("AirCP: cannot recognize the format of indicator Tensor!")
        elif type(omega) == np.ndarray:
            self.Omega = pyten.tenclass.Tensor(omega)
        else:
            self.Omega = omega

        if not self.Omega:
            raise ValueError("AirCP: indicator Tensor cannot be empty!")

        if type(rank) == list or type(rank) == tuple:
            rank = rank[0]


        self.ndims = self.T.ndims
        self.shape = self.T.shape

        if simMats is None:
            self.simMats = np.array([np.identity(self.shape[i]) for i in range(self.ndims)])
        elif type(simMats) != np.ndarray and type(simMats) != list:
            raise ValueError("AirCP: cannot recognize the format of similarity matrices from auxiliary information!")
        else:
            self.simMats = np.array(simMats)
        self.L = np.array([csgraph.laplacian(simMat, normed=False) for simMat in self.simMats])

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

        if printitn == 0:
            printitn = maxIter

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
        self.printitn = printitn

    def initializeLatentMatrices(self):
        self.U = [np.random.rand(self.shape[i], self.rank) for i in range(self.ndims)]
        self.Y = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
        self.Z = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
        self.II = pyten.tools.tendiag(np.ones(self.rank), [self.rank for i in range(self.ndims)])
        self.X = self.T.data + (1-self.Omega.data)*(self.T.norm()/self.T.size())
        self.X = pyten.tenclass.Tensor(self.X)
        self.X_pre = self.X.copy()


    def run(self):
        self.errList = []

        self.initializeLatentMatrices()

        for k in range(self.maxIter):

            # update step eta
            self.eta *= self.rho

            # update Z
            for i in range(self.ndims):
                temp_1 = self.eta*self.U[i] - self.Y[i]
                temp_2 = self.eta*np.identity(self.shape[i]) + self.alpha[i]*self.L[i]
                self.Z[i] = np.dot(np.linalg.inv(temp_2 + 0.00001*np.identity(self.shape[i])), temp_1)

            temp_1, temp_2 = None, None

            # update U
            for i in range(self.ndims):
                # calculate intermedian Tensor and its mode-n unfolding
                midT = self.II.copy()
                # calculate Kronecker product of U(1), ..., U(i-1),U(i+1), ...,U(n)
                for j in range(self.ndims):
                    if j == i:
                        continue
                    midT=midT.ttm(self.U[j], j+1)
                unfoldD_temp = pyten.tenclass.Tenmat(midT, i+1)

                temp_Z = self.eta*self.Z[i] + self.Y[i]
                temp_B = np.dot(unfoldD_temp.data,unfoldD_temp.data.T)
                temp_B += self.eta*np.identity(self.rank) + self.lmbda*np.identity(self.rank)
                temp_B += 0.00001*np.identity(self.rank)
                temp_C = pyten.tenclass.Tenmat(self.X, i+1)
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
                self.Y[i] = self.Y[i] + self.eta * (self.Z[i] - self.U[i])

            # checking the stop criteria
            error = np.linalg.norm(self.X_pre.data - self.X.data)/self.normT
            self.X_pre = self.X.copy()
            self.errList.append(error)

            if (k+1) % self.printitn == 0:
                print 'AirCP: iterations={0}, difference={1}'.format(k+1, self.errList[-1])
            elif error < self.tol:
                print 'AirCP: iterations={0}, difference={1}'.format(k+1, self.errList[-1])

            if error < self.tol:
                break
