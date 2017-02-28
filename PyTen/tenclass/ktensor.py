#!/usr/bin/env python
__author__ = "Hancheng Ge"
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
from PyTen.tenclass import tensor
from PyTen.tools import khatrirao


class ktensor(object):
	"""
	Tensor stored in decomposed form as a Kruskal operator.
	----------
	Intended Usage
		The Kruskal operator is particularly useful to store
		the results of a CP decomposition.
	Parameters
	----------
	lmbda : array_like of floats, optional
			Weights for each dimension of the Kruskal operator.
			``len(lambda)`` must be equal to ``U[i].shape[1]``

	Us : list of ndarrays
		 Factor matrices from which the tensor representation
		 is created. All factor matrices ``U[i]`` must have the
		 same number of columns, but can have different
		 number of rows.
	--------
	"""

	def __init__(self, lmbda = None, Us = None):
		if Us is None:
			raise ValueError("Ktensor: first argument cannot be empty.")
		else:
			self.Us = np.array(Us)
		self.shape = tuple(Ui.shape[0] for Ui in Us)
		self.ndim = len(self.Us)
		self.rank = self.Us[0].shape[1]
		if lmbda is None:
			self.lmbda = np.ones(len(self.rank))
		else:
			self.lmbda = np.array(lmbda)
		if not all(np.array([Ui.shape[1] for Ui in Us]) == self.rank):
			raise ValueError('Ktensor: dimension mismatch of factor matrices')

	def norm(self):
		"""
		Efficient computation of the Frobenius norm for ktensors
		Returns: None
		-------
		norm : float
			Frobenius norm of the ktensor
		"""
		coefMatrix = np.dot(self.Us[0].T,self.Us[0])
		for i in range(1,self.ndim):
			coefMatrix = coefMatrix * np.dot(self.Us[i].T,self.Us[i])
		coefMatrix=np.dot(np.dot(self.lmbda.T,coefMatrix),self.lmbda)
		return np.sqrt(coefMatrix.sum());

	def tondarray(self):
		"""
		Converts a ktensor into a dense multidimensional ndarray

		Returns: None
		-------
		arr : np.ndarray
			  Fully computed multidimensional array whose shape matches
			  the original ktensor.
		"""
		A = np.dot(self.lmbda.T, khatrirao.khatrirao(self.Us).T)
		return A.reshape(self.shape)

	def totensor(self):
		"""
		Converts a ktensor into a dense tensor
		Returns
		-------
		arr : tensor
			  Fully computed multidimensional array whose shape matches
			  the original ktensor.
		"""
		return tensor.tensor(self.tondarray())





