__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"


import numpy
from pyten.tools import khatrirao
import pyten.tenclass


def cp_als(Y, R=20, Omega=None, tol=1e-4, maxiter=100, init='random', printitn=1):
    """ CP_ALS Compute a CP decomposition of a Tensor (and recover it). """
    #   'tol' - Tolerance on difference in fit {1.0e-4}
    #   'maxiters' - Maximum number of iterations {50}
    #   'init' - Initial guess [{'random'}|'nvecs'|cell array]
    #   'printitn' - Print fit every n iterations; 0 for no printing {1}

    X=Y.data.copy()
    X=pyten.tenclass.Tensor(X)
    ##Setting Parameters
    # Construct Omega if no input
    if Omega==None:
        Omega=X.data*0+1

    # Extract number of dimensions and norm of X.
    N = X.ndims
    normX = X.norm()
    dimorder=range(N)  #'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}


    # Define convergence tolerance & maximum iteration
    fitchangetol = tol
    maxiters = maxiter

    # Recover or just decomposition
    recover=0
    if 0 in Omega:
        recover=1


    #   Set up and error checking on initial guess for U.
    if type(init)==list:
        Uinit = init[:]
        if len(Uinit) != N:
            raise IndexError('OPTS.init does not have %d lists', N)
        for n in dimorder[1:]:
            if Uinit[n].shape!=(X.shape[n],R):
                raise IndexError('OPTS.init{%d} is the wrong size', n)
    else:
        # Observe that we don't need to calculate an initial guess for the
        # first index in dimorder because that will be solved for in the first
        # inner iteration.
        if init=='random':
            Uinit = range(N);Uinit[0]=[]
            for n in dimorder[1:]:
                Uinit[n] = numpy.random.random([X.shape[n],R])
        elif init=='nvecs' or init=='eigs':
            Uinit = range(N);Uinit[0]=[]
            for n in dimorder[1:]:
                Uinit[n] = X.nvecs(n,R) #first R leading eigenvecters
        else:
            raise TypeError('The selected initialization method is not supported')

        # Set up for iterations - initializing U and the fit.
        U = Uinit[:]
        fit = 0

        if printitn>0:
            print('\nCP_ALS:\n')

    #Save hadamard product of each U[n].T*U[n]
    UtU = numpy.zeros([N,R,R])
    for n in range(N):
        if len(U[n]):
            UtU[n,:,:] = numpy.dot(U[n].T,U[n])

    for iter in range(1,maxiters+1):
        fitold = fit
        oldX=X.data*1.0

        # Iterate over all N modes of the Tensor
        for n in range(N):
            # Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            temp1=[n]
            temp2=range(n);temp3=range(n+1,N);temp2.reverse();temp3.reverse()
            temp1[len(temp1):len(temp1)]=temp3;temp1[len(temp1):len(temp1)]=temp2
            Xn=X.permute(temp1)
            Xn=Xn.tondarray()
            Xn=Xn.reshape([Xn.shape[0],Xn.size/Xn.shape[0]])
            tempU=U[:];tempU.pop(n);tempU.reverse()
            Unew=Xn.dot(khatrirao(tempU))

            # Compute the matrix of coefficients for linear system
            temp=range(n);temp[len(temp):len(temp)]=range(n+1,N)
            Y = numpy.prod(UtU[temp,:,:],axis=0)
            Unew = Unew.dot(numpy.linalg.inv(Y))

            # Normalize each vector to prevent singularities in coefmatrix
            if (iter == 1):
                lamb = numpy.sqrt(numpy.sum(numpy.square(Unew),0)) #2-norm
            else:
                lamb=numpy.max(Unew,0)
                lamb = numpy.max([lamb,numpy.ones(R)],0) #max-norm

            lamb=[x*1.0 for x in lamb]
            Unew = Unew/numpy.array(lamb)
            U[n] = Unew
            UtU[n,:,:] = numpy.dot(U[n].T,U[n])

        # Reconstructed fitted Ktensor
        P = pyten.tenclass.Ktensor(lamb,U)
        if recover==0:
            if normX == 0:
                fit = P.norm()**2 - 2 * numpy.sum(X.tondarray()*P.tondarray())
            else:
                normresidual = numpy.sqrt( abs(normX**2 + P.norm()**2 - 2 *  numpy.sum(X.tondarray()*P.tondarray())) )
                fit = 1 - (normresidual / normX) #fraction explained by model
                fitchange = abs(fitold - fit)
        else:
            temp=P.tondarray()
            X.data[Omega==0]=temp[Omega==0]
            fitchange=numpy.linalg.norm(X.data-oldX)


        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            flag = 0
        else:
            flag = 1

        if (printitn!=0 and iter%printitn==0) or ((printitn>0) and (flag==0)):
            if recover==0:
                print(' Iter %2d: f = %e f-delta = %7.1e\n', iter, fit, fitchange)
            else:
                print(' Iter %2d: f = %e f-delta = %7.1e\n', iter, fitchange)

        # Check for convergence
        if (flag == 0):
            break

    return P,X
    #T: Decompose result.(kensor)
    #X: Recovered Tensor.












