__author__ = 'Song'

#!/usr/bin/env python
__author__ = "Qingquan Song"
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
import copy
from scipy.sparse import csgraph
from PyTen.tenclass import tensor, tenmat
from PyTen.tools import tendiag, khatrirao
from PyTen.method import cp_als

class OLSGD(object):
    # This routine solves the online tensor decomposition problem using CP decomposition
    # Online tensor completion with CP decomposition (SGD optimization)
    def __init__(self, rank=20, mu=0.01,lmbda=0.1):
        #Initialization stage of OLSTEC
        self.mu = mu
        self.rank = rank
        self.lmbda=lmbda
        self.A=[]
        self.B=[]
        self.X=[]
        self.timestamp=0

    def update(self,newX, omega=None, rank=20, mut=0.01,lmbdat=0.1):
        ## Update stage of OLSGD
        # input:  newX, the new incoming data tensor
        #         As, a cell array contains the previous loading matrices of initX
        # ouputs: As, a cell array contains the updated loading matrices of initX.
        #             To save time, As(N) is not modified, instead, projection of
        #             newX on time mode (alpha) is given in the output
        #         Ps, Qs, cell arrays contain the updated complementary matrices
        #         alpha, coefficient on time mode of newX
        self.mu = mut
        self.rank = rank
        self.lmbda=lmbdat

        if type(newX) != tensor.tensor and type(newX) != np.ndarray:
            raise ValueError("OLSGD: cannot recognize the format of observed tensor!")
        elif type(newX) == np.ndarray:
            self.T = tensor.tensor(newX)
        else:
            self.T = newX

        if omega==None:
            self.omega=self.T.data*0+1
        if type(omega) != tensor.tensor and type(omega) != np.ndarray:
            raise ValueError("OLSGD: cannot recognize the format of indicator tensor!")
        elif type(omega) == np.ndarray:
            self.Omega = tensor.tensor(omega)
        else:
            self.Omega = omega

        dims = list(newX.shape)
        if len(dims)==2:
            dims.insert(0,1)
        self.T.data=self.T.data.reshape(dims)
        self.Omega.data=self.Omega.data.reshape(dims)

        if self.A==[]:
            self.shape=copy.deepcopy(dims)
            self.A = np.random.rand(self.shape[1], self.rank)
            self.B = np.random.rand(self.shape[2], self.rank)
            self.C = np.zeros([self.shape[0], self.rank])
        else:
            self.shape[0]+=dims[0]
            self.C=np.row_stack((self.C,np.zeros([dims[0], self.rank])))

        for i in range(dims[0]):
            self.timestamp+=1
            Omg=self.Omega.data[i,:,:]
            index=Omg.nonzero()
            m=index[0]
            n=index[1]
            NNZ=len(m)
            temp1=self.lmbda*np.identity(self.rank)
            temp2=np.zeros([self.rank,1])
            for j in range(NNZ):
                temp3=self.A[m[j],:]*self.B[n[j],:]
                temp3=temp3.reshape([temp3.shape[0],1])
                temp1+=temp3.dot(temp3.T)
                temp2+=self.T.data[i,m[j],n[j]]*temp3
            temp=np.dot(np.linalg.inv(temp1),temp2)
            self.C[self.timestamp-1,:]=temp.T;
            tempA=copy.deepcopy(self.A)
            temp=temp.reshape(temp.shape[0]);
            Err=Omg*(self.T.data[i,:,:]-np.dot(np.dot(self.A,np.diag(temp)),self.B.T))
            self.A=(1-self.lmbda/self.timestamp/self.mu)*self.A+\
                   1/self.mu*np.dot(np.dot(Err,self.B),np.diag(temp))
            self.B=(1-self.lmbda/self.timestamp/self.mu)*self.B+\
                   1/self.mu*np.dot(np.dot(Err.T,tempA),np.diag(temp))


            fitX=np.dot(np.dot(self.A,np.diag(temp)),self.B.T)
            fitX=fitX.reshape([1,dims[1],dims[2]])
            if self.X==[]:
                self.X=tensor.tensor(fitX)
                recX=self.T.data[0,:,:] * self.Omega.data[0,:,:] + self.X.data * (1 - self.Omega.data[0,:,:])
                self.Rec=tensor.tensor(recX)
            else:
                self.X=tensor.tensor(np.row_stack((self.X.data,fitX)))
                recX=self.T.data[i,:,:] * self.Omega.data[i,:,:] + fitX * (1 - self.Omega.data[i,:,:])
                self.Rec=tensor.tensor(np.row_stack((self.Rec.data,recX)))