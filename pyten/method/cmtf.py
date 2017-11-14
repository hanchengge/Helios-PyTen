__author__ = 'Song'

__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import numpy

from pyten.tools import tools
import pyten.tenclass
import pyten.method


def cmtf(X, Y=None, CM=None, R=2, Omega=None, tol=1e-4, maxiter=100, init='random', printitn=100):
    """ CMTF Compute a Coupled Matrix and Tensor Factorization (and recover the Tensor).
    #   'X' - Tensor
    #   'Y' - Coupled Matries
    #   'CM' - Shared Modes
    #   'tol' - Tolerance on difference in fit {1.0e-4}
    #   'maxiters' - Maximum number of iterations {50}
    #   'init' - Initial guess [{'random'}|'nvecs'|cell array]
    #   'printitn' - Print fit every n iterations; 0 for no printing {1}"""

    ##Setting Parameters
    if type(R) == list or type(R) == tuple:
        R = R[0]

    if Y is None:
        [P, X] = pyten.method.cp_als(X, R, Omega, tol, maxiter, init, printitn)
        V = None
        return P, X, V

    if CM is None:
        CM = 0
    elif int == type(CM):
        CM = CM - 1
    else:
        CM = [i - 1 for i in CM]

    # Construct Omega if no input
    if Omega is None:
        Omega = X.data * 0 + 1

    # Extract number of dimensions and norm of X.
    N = X.ndims
    normX = X.norm()
    dimorder = range(N)  # 'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}


    # Define convergence tolerance & maximum iteration
    fitchangetol = 1e-4
    maxiters = maxiter

    # Recover or just decomposition
    recover = 0
    if 0 in Omega:
        recover = 1

    # Set up and error checking on initial guess for U.
    if type(init) == list:
        Uinit = init[:]
        if len(Uinit) != N:
            raise IndexError('OPTS.init does not have %d lists', N)
        for n in dimorder[1:]:
            if Uinit[n].shape != (X.shape[n], R):
                raise IndexError('OPTS.init{%d} is the wrong size', n)
    else:
        # Observe that we don't need to calculate an initial guess for the
        # first index in dimorder because that will be solved for in the first
        # inner iteration.
        if init == 'random':
            Uinit = range(N)
            Uinit[0] = []
            for n in dimorder[1:]:
                Uinit[n] = numpy.random.random([X.shape[n], R])
        elif init == 'nvecs' or init == 'eigs':
            Uinit = range(N)
            Uinit[0] = []
            for n in dimorder[1:]:
                Uinit[n] = X.nvecs(n, R)  # first R leading eigenvecters
        else:
            raise TypeError('The selected initialization method is not supported')

        # Set up for iterations - initializing U and the fit.
        U = Uinit[:]
        if type(CM) == int:
            V = numpy.random.random([Y.shape[1], R])
        else:
            V = [numpy.random.random([Y[i].shape[1], R]) for i in range(len(CM))]
        fit = 0

        if printitn > 0:
            print('\nCMTF:\n')

    # Save hadamard product of each U[n].T*U[n]
    UtU = numpy.zeros([N, R, R])
    for n in range(N):
        if len(U[n]):
            UtU[n, :, :] = numpy.dot(U[n].T, U[n])

    for iter in range(1, maxiters + 1):
        fitold = fit
        oldX = X.data * 1.0

        # Iterate over all N modes of the Tensor
        for n in range(N):
            # Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            temp1 = [n]
            temp2 = range(n)
            temp3 = range(n + 1, N)
            temp2.reverse()
            temp3.reverse()
            temp1[len(temp1):len(temp1)] = temp3
            temp1[len(temp1):len(temp1)] = temp2
            Xn = X.permute(temp1)
            Xn = Xn.tondarray()
            Xn = Xn.reshape([Xn.shape[0], Xn.size / Xn.shape[0]])
            tempU = U[:]
            tempU.pop(n)
            tempU.reverse()
            Unew = Xn.dot(pyten.tools.khatrirao(tempU))

            # Compute the matrix of coefficients for linear system
            temp = range(n)
            temp[len(temp):len(temp)] = range(n + 1, N)
            B = numpy.prod(UtU[temp, :, :], axis=0)
            if int != type(CM):
                tempCM = [i for i, a in enumerate(CM) if a == n]
            elif CM == n:
                tempCM = [0]
            else:
                tempCM = []
            if tempCM != [] and int != type(CM):
                for i in tempCM:
                    B = B + numpy.dot(V[i].T, V[i])
                    Unew = Unew + numpy.dot(Y[i], V[i])
                    V[i] = numpy.dot(Y[i].T, Unew)
                    V[i] = V[i].dot(numpy.linalg.inv(numpy.dot(Unew.T, Unew)))
            elif tempCM != []:
                B = B + numpy.dot(V.T, V)
                Unew = Unew + numpy.dot(Y, V)
                V = numpy.dot(Y.T, Unew)
                V = V.dot(numpy.linalg.inv(numpy.dot(Unew.T, Unew)))
            Unew = Unew.dot(numpy.linalg.inv(B))
            U[n] = Unew
            UtU[n, :, :] = numpy.dot(U[n].T, U[n])

        # Reconstructed fitted Ktensor
        lamb = numpy.ones(R)
        P = pyten.tenclass.Ktensor(lamb, U)
        if recover == 0:
            if normX == 0:
                fit = P.norm() ** 2 - 2 * numpy.sum(X.tondarray() * P.tondarray())
            else:
                normresidual = numpy.sqrt(normX ** 2 + P.norm() ** 2 - 2 * numpy.sum(X.tondarray() * P.tondarray()))
                fit = 1 - (normresidual / normX)  # fraction explained by model
                fitchange = abs(fitold - fit)
        else:
            temp = P.tondarray()
            X.data = temp * (1-Omega) + X.data * Omega
            fitchange = numpy.linalg.norm(X.data - oldX)


        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            flag = 0
        else:
            flag = 1

        if (printitn != 0 and iter % printitn == 0) or ((printitn > 0) and (flag == 0)):
            if recover == 0:
                print 'CMTF: iterations={0}, f={1}, f-delta={2}'.format(iter, fit, fitchange)
            else:
                print 'CMTF: iterations={0}, f-delta={1}'.format(iter, fitchange)
        if (flag == 0):
            break

    return P, X, V
    # P: Decompose result.(kensor)
    # X: Recovered Tensor.
    # V: Projection Matrix.
