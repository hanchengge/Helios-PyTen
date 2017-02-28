___author__ = 'Hancheng Ge'

from PyTen.tenclass import tensor
import numpy as np
import numpy.matlib 


def tendiag(v, sz):
	# Make sure v is a column vector
	v = np.array(v)
	N = v.size
	v = v.reshape((N,1))

	X = np.zeros(sz)

	subs = np.matlib.repmat(np.array(range(N)).reshape(N,1), 1, len(sz))

	for i in range(N):
		X[subs[i][0],subs[i][1],subs[i][2]] = v[i]

	return tensor.tensor(X)
