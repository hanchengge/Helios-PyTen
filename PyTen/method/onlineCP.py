#!/usr/bin/env python
__author__ = "Qingquan Song"
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
from scipy.sparse import csgraph
from PyTen.tenclass import tensor, tenmat
from PyTen.tools import tendiag, khatrirao
from PyTen.method import cp_als

class onlineCP(object):
    """  Accelerating Online CP Decompositions for Higher Order Tensors. """
    # This routine solves the online tensor decomposition problem using CP decomposition

    def __init__(self, initX, rank=20, tol=1e-8):
        #Initialization stage of OnlineCP
        #input:  initX, data tensor used for initialization
        #         As, a cell array contains the loading matrices of initX
        #         R, tensor rank
        # ouputs: Ps, Qs, cell arrays contain the complementary matrices

        #function [ Ps, Qs ] = onlineCP_initial( initX, As, R )
        #dims = size(initX);
        #N = length(dims);
        #if As is not given, calculate the CP decomposition of the initial data

        if not initX:
            raise ValueError("OnlineCP: Initial tensor cannot be empty!")
        elif type(initX) != tensor.tensor and type(initX) != np.ndarray:
            raise ValueError("OnlineCP: cannot recognize the format of observed tensor!")
        elif type(initX) == np.ndarray:
            self.T = tensor.tensor(initX)
        else:
            self.T = initX

        self.ndims = self.T.ndims
        self.shape = self.T.shape

        [K,Rec] = cp_als.cp_als(self.T, rank, None, tol)
        # absorb lambda into the last dimension
        self.As=K.Us
        self.As[self.ndims-1] = self.As[self.ndims-1]*K.lmbda;

        # for the first N-1 modes, calculte their assistant matrices P and Q
        #H = getHadamard(As);
        #Ks = getKhatriRaoList((As(1:N-1)));
        AtA = np.zeros([self.ndims,rank,rank]);
        for n in range(self.ndims):
            if len(self.As[n]):
                AtA[n,:,:] = np.dot(self.As[n].T,self.As[n]);

        self.Ps = range(self.ndims-1);
        self.Qs = range(self.ndims-1);
        for n in range(self.ndims-1):
            #Xn = reshape(permute(initX, [n, 1:n-1, n+1:N]), dims(n), []);
            #Ps{n} = Xn*khatrirao(As{N}, Ks{n});
            #Qs{n} = H./(As{n}'*As{n});
            temp1=[n];
            temp2=range(n);temp3=range(n+1,self.ndims);temp2.reverse();temp3.reverse()
            temp1[len(temp1):len(temp1)]=temp3;temp1[len(temp1):len(temp1)]=temp2
            Xn=self.T.permute(temp1)
            Xn=Xn.tondarray()
            Xn=Xn.reshape([Xn.shape[0],Xn.size/Xn.shape[0]])
            tempAs=range(self.ndims);
            for i in range(self.ndims):
                tempAs[i]=self.As[i];
            tempAs.pop(n);tempAs.reverse()
            self.Ps[n]=Xn.dot(khatrirao.khatrirao(tempAs))
            temp=range(n);temp[len(temp):len(temp)]=range(n+1,self.ndims)
            self.Qs[n] = np.prod(AtA[temp,:,:],axis=0)


        self.rank = rank
        #self.maxIter = maxIter
        self.tol = tol
        self.X = tendiag.tendiag(np.ones(self.rank), [self.rank for i in range(self.ndims)])
        for i in range(self.ndims):
            self.X=self.X.ttm(self.As[i], i+1)

#    def initializeLatentMatrices(self):
#        self.U = [np.random.rand(self.shape[i], self.rank) for i in range(self.ndims)]
#        self.Y = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
#        self.Z = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
#        self.II = tendiag.tendiag(np.ones(self.rank), [self.rank for i in range(self.ndims)])
#        self.X = self.T.data + (1-self.Omega.data)*(self.T.norm()/self.T.size())
#        self.X = tensor.tensor(self.X)
#        self.X_pre = self.X.copy()

    def getKhatriRaoList(self):

        lefts = self.As[self.ndims-2]
        rights = self.As[0]
        if self.ndims>3:
            for n in range(1,self.ndims-1):
                lefts = [lefts,khatrirao.khatrirao(lefts[n-1],self.As[self.ndims-n] )]
                rights = [rights,khatrirao.khatrirao(self.As[n], rights[n-1])]

        self.Ks=range(self.ndims-1)

        if self.ndims>3:
            self.Ks[0] = lefts[self.ndims-3]
            self.Ks[self.ndims-2] = rights[self.ndims-3]
            for n in range(1,self.ndims-1):
                self.Ks[n] = khatrirao.khatrirao(lefts[self.ndims-n-1], rights[n-1])
        else:
            self.Ks[0]=lefts
            self.Ks[self.ndims-2]= rights


    def getHadamard(self):
        self.H = None
        for n in range(self.ndims-1):
            if self.H==None:
                self.H = np.dot(self.As[n].T,self.As[n])
            else:
                self.H = self.H*(np.dot(self.As[n].T,self.As[n]))


    def update(self,newX):
        ## Update stage of OnlineCP
        # input:  newX, the new incoming data tensor
        #         As, a cell array contains the previous loading matrices of initX
        #         Ps, Qs, cell arrays contain the previous complementary matrices
        # ouputs: As, a cell array contains the updated loading matrices of initX.
        #             To save time, As(N) is not modified, instead, projection of
        #             newX on time mode (alpha) is given in the output
        #         Ps, Qs, cell arrays contain the updated complementary matrices
        #         alpha, coefficient on time mode of newX

        #function [ As, Ps, Qs, alpha ] = onlineCP_update( newX, As, Ps, Qs )
        dims = list(newX.shape);
        if len(dims)==self.ndims-1:
            dims.append(1)

        batchSize = dims[self.ndims-1];

        self.getKhatriRaoList();
        self.getHadamard();

        # update mode-N
        KN = khatrirao.khatrirao([self.Ks[0], self.As[0]]);

        n=self.ndims-1;
        temp1=[n];
        temp2=range(n);temp2.reverse();
        temp1[len(temp1):len(temp1)]=temp2;
        newXN=newX.permute(temp1)
        newXN=newXN.tondarray();
        newXN=newXN.reshape([newXN.shape[0],newXN.size/newXN.shape[0]]);
        self.alpha = np.dot(newXN.dot(KN),np.linalg.inv(self.H));

        # update mode 1 to N-1
        for n in range(self.ndims-1):
            temp1=[n];
            temp2=range(n);temp3=range(n+1,self.ndims);temp2.reverse();temp3.reverse();
            temp1[len(temp1):len(temp1)]=temp3;temp1[len(temp1):len(temp1)]=temp2;
            newXn=newX.permute(temp1);
            newXn=newXn.tondarray();
            newXn=newXn.reshape([newXn.shape[0],newXn.size/newXn.shape[0]]);

            self.Ps[n] = self.Ps[n]+newXn.dot(khatrirao.khatrirao([self.alpha, self.Ks[n]]));
            Hn = self.H/np.dot(self.As[n].T,self.As[n]);
            self.Qs[n] = self.Qs[n]+(np.dot(self.alpha.T,self.alpha))*Hn;
            self.As[n] = self.Ps[n].dot(np.linalg.inv(self.Qs[n]));

        self.As[self.ndims-1]=np.row_stack((self.As[self.ndims-1],self.alpha))
        self.X = tendiag.tendiag(np.ones(self.rank), [self.rank for i in range(self.ndims)])
        for i in range(self.ndims):
            self.X=self.X.ttm(self.As[i], i+1)