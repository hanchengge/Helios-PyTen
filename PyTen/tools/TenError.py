#!/usr/bin/env python
__author__ = "Qingquan Song"
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
from PyTen.tenclass import tensor

def TenError(fitX,realX,Omega):
    """Calculate Three Kinds of Error"""
    if type(Omega)!=np.ndarray and type(Omega)!=tensor.tensor:
        raise ValueError("AirCP: cannot recognize the format of observed tensor!")
    elif type(Omega) == tensor.tensor:
        Omega=Omega.tondarray
    Norm1=np.linalg.norm(realX.data)
    Norm2=np.linalg.norm(realX.data*(1-Omega))
    Err1=np.linalg.norm(fitX.data-realX.data)
    Err2=np.linalg.norm((fitX.data-realX.data)*(1-Omega))
    ReErr1=Err1/Norm1
    ReErr2=Err2/Norm2
    return Err1,ReErr1,ReErr2
