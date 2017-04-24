#!/usr/bin/env python
__author__ = "Hancheng Ge"
__copyright__ = "Copyright 2016, The Helios Project"

import sys

sys.path.append('/path/to/pyten/tenclass')
import sptensor
import numpy as np
from pyten.tools import tools


class Tensor(object):
    def __init__(self, data=None, shape=None):
        # constructor for Tensor object.
        # data can be numpy.array or list.
        # shape can be tuple, numpy.array, or list of integers
        if data is None:
            raise ValueError('Tensor: first argument cannot be empty.')

        if data.__class__ == list or data.__class__ == np.ndarray:
            if data.__class__ == list:
                data = np.array(data)
        else:
            raise ValueError('Tensor: first argument should be either list or numpy.ndarray')

        if shape:
            if shape.__class__ != tuple:
                if shape.__class__ == list:
                    if (len(shape) < 2):
                        raise ValueError('Tensor: second argument must be a row vector with at least two elements.')
                    elif shape[0].__class__ != int:
                        if len(shape[0]) == 1:
                            shape = [y for x in shape for y in x]
                        else:
                            raise ValueError('Tensor: second argument must be a row vector with integers.')
                    else:
                        shape = tuple(shape)
                elif shape.__class__ == np.ndarray:
                    if shape.ndim != 2:
                        raise ValueError('Tensor: second argument must be a row vector with integers.')
                    elif shape[0].__class__ != np.int64:
                        if len(shape[0]) == 1:
                            shape = [y for x in shape for y in x]
                        else:
                            raise ValueError('Tensor: second argument must be a row vector with integers.')
                    else:
                        shape = tuple(shape)
                else:
                    raise ValueError('Tensor: second argument must be a row vector (tuple).')
            else:
                if len(shape) < 2:
                    raise ValueError('Tensor: second argument must be a row vector with at least two elements.')
                elif len(shape[0]) != 1:
                    raise ValueError('Tensor: second argument must be a row vector.')
            if tools.prod(shape) != data.size:
                raise ValueError("Tensor: size of data does not match specified size of Tensor.")
        else:
            shape = tuple(data.shape)

        self.shape = shape
        self.data = data.reshape(shape, order='F')
        self.ndims = len(self.shape)

    def __str__(self):
        string = "Tensor of size {0} with {1} elements.\n".format(self.shape, tools.prod(self.shape))
        return string

    def size(self):
        # returns the number of elements in the Tensor
        return self.data.size

    def copy(self):
        # returns a deepcpoy of Tensor object
        return Tensor(self.data)

    def dimsize(self, idx=None):
        # returns the size of the specified dimension
        if idx is None:
            raise ValueError('Please specify the index of that dimension.')
        if (idx.__class__ != int):
            raise ValueError('Index of the dimension must be an integer.')
        if (idx >= self.ndims):
            raise ValueError('Index exceeds the number of dimensions.')
        return self.shape[idx]

    def tosptensor(self):
        """ returns the Sptensor object
            that contains the same value with the Tensor object."""

        length = len(self.shape)
        sub = tools.allIndices(self.shape)
        return sptensor.Sptensor(
            sub,
            self.data.flatten().reshape(self.data.size, 1),
            self.shape)

    def permute(self, order=None):
        """ returns a Tensor permuted by the order specified."""
        if order is None:
            raise ValueError("Permute: Order must be specified.")

        if order.__class__ == list or order.__class__ == tuple:
            order = np.array(order)

        if self.ndims != len(order):
            raise ValueError("Permute: Invalid permutation order.")

        if not (sorted(order) == np.arange(self.ndims)).all():
            raise ValueError("Permute: Invalid permutation order.")

        newdata = self.data.copy()

        newdata = newdata.transpose(order)

        return Tensor(newdata)

    def ipermute(self, order=None):
        # returns a Tensor permuted by the inverse of the order specified
        if (order is None):
            raise ValueError('Please specify the order.')
            raise ValueError('Ipermute: please specify the order.')

        iorder = [order[idx] for idx in range(0, len(order))]
        if order.__class__ == np.array or order.__class__ == tuple:
            order = list(order)
        else:
            if order.__class__ != list:
                raise ValueError('Ipermute: permutation order must be a list.')

        if not self.ndims == len(order):
            raise ValueError("Ipermute: invalid permutation order.")
        if not ((sorted(order) == np.arange(self.ndims)).all()):
            raise ValueError("Ipermute: invalid permutation order.")

        iorder = [order.index(idx) for idx in range(0, len(order))]

        return self.permute(iorder)

    def tondarray(self):
        # returns data of the Tensor with a numpy.ndarray object
        return self.data

    def ttm(self, mat=None, mode=None, option=None):
        # multiplies the Tensor with the given matrix.
        # the given matrix is a single 2-D array with list or numpy.array.
        if mat is None:
            raise ValueError('Tensor/TTM: matrix (mat) needs to be specified.')

        if mode is None or mode.__class__ != int or mode > self.ndims or mode < 1:
            raise ValueError('Tensor/TTM: mode must be between 1 and NDIMS(Tensor).')

        if mat.__class__ == list:
            matrix = np.array(mat)
        elif mat.__class__ == np.ndarray:
            matrix = mat
        else:
            raise ValueError('Tensor/TTM: matrix must be a list or a numpy.ndarray.')

        if len(matrix.shape) != 2:
            raise ValueError('Tensor/TTM: first argument must be a matrix.')

        if (matrix.shape[1] != self.shape[mode - 1]):
            raise ValueError('Tensor/TTM: matrix dimensions must agree.')

        dim = mode - 1
        N = self.ndims
        shape = list(self.shape)
        order = [dim] + range(0, dim) + range(dim + 1, N)
        newData = self.permute(order).data
        newData = newData.reshape(shape[dim], tools.prod(shape) / shape[dim])
        if option is None:
            newData = np.dot(matrix, newData)
            p = matrix.shape[0]
        elif (option == 't'):
            newData = np.dot(matrix.transpose(), newData)
            p = matrix.shape[1]
        else:
            raise ValueError('Tensor/TTM: unknown option')
        newShape = [p] + shape[0:dim] + shape[dim + 1:N]
        newData = Tensor(newData.reshape(newShape))
        newData = newData.ipermute(order)

        return newData

    def ttv(self, vec=None, mode=None, opt=None):
        # multiplies the Tensor with the given vector.
        # the given vector is a single 1-D array with list or numpy.array.
        if (vec is None):
            raise ValueError('Tensor/TTV: vector (vec) needs to be specified.')

        if mode is None or mode.__class__ != int or mode > self.ndims or mode < 1:
            raise ValueError('Tensor/TTM: mode must be between 1 and NDIMS(Tensor).')

        if vec.__class__ == list:
            vector = np.array(vec)
        elif vec.__class__ == np.ndarray:
            vector = vec
        else:
            raise ValueError('Tensor/TTV: vector must be a list or a numpy.ndarray.')

        if len(vector.shape) != 1:
            raise ValueError('Tensor/TTV: first argument must be a vector.')

        if vector.shape[0] != self.shape[mode - 1]:
            raise ValueError('Tensor/TTV: vector dimension must agree.')

        dim = mode - 1
        N = self.ndims
        shape = list(self.shape)
        order = [dim] + range(0, dim) + range(dim + 1, N)
        newData = self.permute(order).data
        newData = newData.reshape(shape[dim], tools.prod(shape) / shape[dim])
        newData = np.dot(vector, newData)
        newShape = [1] + shape[0:dim] + shape[dim + 1:N]
        newData = Tensor(newData.reshape(newShape))
        newData = newData.ipermute(order)

        return newData

    def norm(self):
        """return an Frobenius norm of the Tensor"""
        return np.linalg.norm(self.data)

    def unfold(self, n):
        N = self.ndims
        temp1 = [n]
        temp2 = range(n)
        temp3 = range(n + 1, N)
        temp1[len(temp1):len(temp1)] = temp2
        temp1[len(temp1):len(temp1)] = temp3
        Xn = self.permute(temp1)
        Xn = Xn.tondarray()
        Xn = Xn.reshape([Xn.shape[0], np.prod(Xn.shape) / Xn.shape[0]])
        return Xn

    def nvecs(self, n, r):
        """return first r eigenvector of the mode-n unfolding matrix"""
        Xn = self.unfold(n)
        [EigenValue, EigenVector] = np.linalg.eig(Xn.dot(Xn.transpose()))
        return EigenVector[:, range(r)]


if __name__ == '__main__':
    X = Tensor(range(1, 25), [2, 4, 3])
    V = [[1, 2], [2, 1]]
    Y = X.ttm(V, 1)
    print Y.data[:, :, 0]
    print X.__str__()
    print X.__class__
