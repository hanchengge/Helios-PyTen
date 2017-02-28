#!/usr/bin/env python
__author__ = "Qingquan Song, Hancheng Ge"
__copyright__ = "Copyright 2016, The Helios Project"


import numpy as np

def khatrirao(U):
    r=U[0].shape[1];
    K=[];
    N = len(U);
    for j in range(r):
        temp=1;
        for i in range(N):
            temp1=np.outer(temp,U[i][:,j]);
            temp=temp1.reshape([1,temp1.size]);
        #K=temp.reshape([len(temp)/r,r]);
        K=np.append(K,temp);
    K=(K.reshape([r,len(K)/r])).transpose();
    return K;