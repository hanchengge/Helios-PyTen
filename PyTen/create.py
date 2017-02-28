__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
import random
#import ktensor
from PyTen.tenclass import tensor,ttensor,ktensor
from PyTen.tools import *

def create(siz,R=2,M=0,tp='CP'):
    """ A function to create a tensor decomposition or completion problem. """
    # siz: size of tensor
    # R: rank of tensor
    # M: missing percentage of data
    # tp: type of expect solution. (Tucker or CP)
    if type(R)==int:
        R=np.zeros(3)+R;

    N=len(siz);
    if tp=='Tucker':
        U = range(N);
        for n in range(N):
            U[n] = np.random.random([siz[n],R[n]]);
        core=tensor.tensor(np.random.random(R));
        sol=ttensor.ttensor(core, U);
        T=sol.totensor();
        Omega = (np.random.random(siz) > M)*1;
        T.data[Omega==0]=T.data[Omega==0]*0;

    elif tp=='CP':
        U = range(N);
        for n in range(N):
            U[n] = np.random.random([siz[n],R[n]]);
        lmbda=np.ones(R[1]);
        sol=ktensor.ktensor(lmbda,U);
        T=sol.totensor();
        Omega = (np.random.random(siz) > M)*1;
        T.data[Omega==0]=T.data[Omega==0]*0;
    else:
        raise ValueError('No Such Method');



    return T,Omega,sol;
    # T: generated tensor;
    # Omega: missing data matrix (0: Miss; 1:Exist);
    # sol: solution.
