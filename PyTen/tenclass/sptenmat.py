#!/usr/bin/env python
__author__ = "Hancheng Ge"
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
from PyTen.tools import tools
from PyTen.tenclass import *
from scipy import sparse

class sptenmat(object):
	def __init__(X = None, rdim = None, cdim = None, tsize = None):
		# create a sptenmat object from a given ndarray or sptensor X
		if X is None:
			raise ValueError('Sptenmat: first argument cannot be empty.')

		if X.__class__ != sptensor.sptensor:
			raise ValueError("Sptenmat: tensor must be a sptensor object.")

		if rdim is None:
			raise ValueError('Sptenmat: second argument cannot be empty.')

		if rdim.__class__ == list or rdim.__class__ == int:
			rdim = np.array(rdim) - 1

		if cdim is None:
			cdim = np.array([x for x in range(0,X.ndims) if x not in rdim])
		elif cdim.__class__ == list or cdim.__class__ == int:
			cdim = np.array(cdim) - 1
		else:
			raise ValueError("Sptenmat: incorrect specification of dimensions.")

		if not (range(0, X.ndims) == sorted(np.append(rdim, cdim))):
			print (range(0, X.ndims) == sorted(np.append(rdim, cdim)))
			raise ValueError("Tenmat: second argument must be a list or an integer.")

		self.shape = X.shape
		rsize = self.shape[rdim.tolist()]
		csize = self.shape[cdim.tolist()]

		rsubs = X.subs[:,rdim.tolist()]
		csubs = X.subs[:,cdim.tolist()]

		ridxs = np.array([sub2ind(rsize, rsub) for rsub in rsubs])
		cidxs = np.array([sub2ind(csize, csub) for csub in csubs])

		self.subs = np.concatenate((ridxs[newaxis].transpose(),cidxs[np.newaxis].transpose()), axis=1)
		self.vals = X.vals
		self.rdim = rdim
		self.cdim = cdim

	def tosptensor(self):
		# returns a sptensor object based on a sptenmat
		rsize = self.shape[self.rdim.tolist()]
		csize = self.shape[self.cdim.tolist()]

		ridxs = self.subs[:,0]
		cidxs = self.subs[:,1]

		rsubs = np.array([ind2sub(rsize, ridx) for ridx in ridx])
		csubs = np.array([ind2sub(csize, cidx) for cidx in cidx])

		newsubs = np.concatenate((rsubs,csubs), axis=1)

		order = np.concatenate((self.rdim, self.cdim), axis=0)

		newsubs = newsubs[:, order]

		return sptensor.sptensor(newsubs, self.vals, self.shape)  

	def __str__(self):
		ret =""
		ret += "sptenmat from an sptensor of size {0} with {1} nonzeros\n".format(self.shape, len(self.vals))
		return ret
