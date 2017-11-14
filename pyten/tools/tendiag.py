___author__ = 'Hancheng Ge'

import pyten.tenclass
import numpy as np
import numpy.matlib 


def tendiag(v, sz):
	"""Create a Diagonal Tensor of Size 'sz' with Diagnal Values 'v' """
	v = np.array(v)
	N = v.size
	v = v.reshape((N,1))
	X = np.zeros(sz)
	subs = np.matlib.repmat(np.array(range(N)).reshape(N,1), 1, len(sz))

	for i in range(N):
		X[subs[i][0],subs[i][1],subs[i][2]] = v[i]

	return pyten.tenclass.Tensor(X)
