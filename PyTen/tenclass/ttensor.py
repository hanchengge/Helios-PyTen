#!/usr/bin/env python
__author__ = "Hancheng Ge"
__copyright__ = "Copyright 2016, The Helios Project"

from PyTen.tenclass import tensor
import numpy as np

class ttensor(object):
    def __init__(self, core, Us):
        # create a tucker tensor object with the core and latent matrices.
        if core.__class__ != tensor.tensor:
            raise ValueError("Ttensor: core must a tensor object.")

        if Us.__class__ != list:
            raise ValueError("Ttensor: latent matrices should be a list of matrices.")
        else:
            for i,U in enumerate(Us):
                if np.array(U).ndim != 2:
                    raise ValueError("Ttensor: latent matrix U{0} must be a 2-D matrix.".format(i))

        if core.ndims != len(Us):
            raise ValueError("Ttensor: number of dimensions of the core tensor is different with number of latent matrices.")

        k = core.shape

        for i in range(0, core.ndims):
            if k[i] != np.array(Us[i]).shape[1]:
                raise ValueError("Ttensor: latent matrix U{0} does not have {1} columns.".format(i, k[i]))

        self.core = core.copy()
        self.us = Us

        shape = []
        for i in range(0, core.ndims):
            shape.append(len(Us[i]))
        self.shape = tuple(shape)
        self.ndims = core.ndims

    def size(self):
        # returns the size of this tucker tensor
        return tools.prod(self.shape)

    def dimsize(self, idx = None):
        # returns the size of the dimension specified by index
        if idx is None:
            raise ValueError("Ttensor: index of a dimension cannot be empty.")
        if idx.__class__ != int or idx < 1 or idx > self.ndims:
            raise ValueError("Ttensor: index of a dimension is an integer between 1 and NDIMS(tensor).")
        idx = idx - 1
        return self.shape[idx]

    def copy(self):
        # returns a deepcpoy of tucker tensor object
        return ttensor(self.core, self.us)

    def totensor(self):
        # returns a tensor object that is represented by the tucker tensor
        X = self.core.copy()
        for i in range(0, self.ndims):
            X=X.ttm(self.us[i], i+1)
        return X

    def __str__(self):
        ret = "ttensor of size {0}\n".format(self.shape);
        ret += "Core = {0} \n".format(self.core.__str__());
        for i in range(0, self.ndims):
            ret += "u[{0}] =\n{1}\n".format(i, self.us[i]);
        return ret;
