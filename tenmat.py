import numpy as np
import tensor
import tools


class tenmat(object):
	def __init__(self, X = None, rdim = None, cdim = None, tsize = None):
		if X is None:
			raise ValueError('Tenmat: first argument cannot be empty.')

		if X.__class__ == tensor.tensor:
			# convert a tensor to a matrix
			if rdim is None:
				raise ValueError('Tenmat: second argument cannot be empty.')

			if rdim.__class__ == list or rdim.__class__ == int:
				rdim = np.array(rdim) - 1

			self.shape = X.shape

			if cdim is None:
				cdim = np.array([x for x in range(0,X.ndims) if x not in rdim])
			elif cdim.__class__ == list or cdim.__class__ == int:
				cdim = np.array(cdim) - 1
			else:
				raise ValueError("Tenmat: incorrect specification of dimensions.")

			if not (range(0, X.ndims) == sorted(np.append(rdim, cdim))):
				raise ValueError("Tenmat: second argument must be a list or an integer.")

			self.rowIndices = rdim
			self.colIndices = cdim
			
			X = X.permute(np.append(rdim,cdim));

			row = tools.prod([self.shape[x] for x in rdim])
			col = tools.prod([self.shape[x] for x in cdim])

			self.data = X.data.reshape([row, col], order='F')
		elif X.__class__ == numpy.ndarray:
			# copy a matrix to a tenmat object
			if len(X.shape) != 2:
				raise ValueError("Tenmat: first argument must be a 2-D numpy array when converting a matrix to tenmat.")

			if tsize is None:
				raise ValueError("Tenmat: tensor size must be specified as a tuple.")
			else:
				if rdim is None or cdim is None or rdim.__class__ != list or cdim.__class__ != list:
					raise ValueError("Tenmat: second and third arguments must be specified with list.")
				else:
					rdim = np.array(rdim) - 1
					cdim = np.array(cdim) - 1
					if tools.prod([tsize[idx] for idx in rdim]) != X.shape[0]:
						raise ValueError("Tenmat: matrix size[0] does not match the tensor size specified.")
					if tools.prod([tsize[idx] for idx in cdim]) != X.shape[1]:
						raise ValueError("Tenmat: matrix size[1] does not match the tensor size specified.")
			self.data = X
			self.rowIndices = rdim
			self.colIndices = cdim
			self.shape = tsize

	def copy(self):
		# returns a deepcpoy of tenmat object
		return tenmat(self.data, self.rowIndices, self.colIndices, self.shape)

	def totensor(self):
		# returns a tensor object based on a tenmat
		order = np.append(self.rowIndices, self.colIndices)
		data = self.data.reshape([self.shape[idx] for idx in order], order='F');
		tData = tensor.tensor(data).ipermute(order);
		return tData

	def tondarray(self):
		# returns data of a tenmat with a numpy.ndarray object
		return self.data

	def __str__(self):
		ret = ""
		ret += "Matrix corresponding to a tensor of size {0}\n".format(self.shape)
		ret += "Row Indices {0}\n".format(self.rowIndices+1)
		ret += "Column Indices {0}\n".format(self.colIndices+1)
		return ret


if __name__ == '__main__':
	X = tensor.tensor(range(1,25),[3,2,2,2])
	print X.data[:,:,0,0]
	A = tenmat(X,[1,2],[4,3])
	print A.data
	print A.totensor().data[:,:,0,0]
	print A.__str__()