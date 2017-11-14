__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
import pyten.tenclass


def tucker_als(Y,R=20,Omega=None,tol=1e-4,maxiter=100,init='random',printitn=100):
    """ TUCKER_ALS Higher-order orthogonal iteration. """
    #Y: Original Tensor
    #R: The rank of the Tensor you want to use for  approximation (recover or decompose).{Default: None}
    #tol: Tolerance on difference in fit.(Convergence tolerance for both cp(als) or tucker(als).){Default: 1.0e-4}
    #maxiter: Maximum number of iterations {Default: 50}
    #init: Initial guess 'random'|'nvecs'|'eigs'. {Default 'random'}
    #printitn: Print fit every n iterations; 0 for no printing.

    X=Y.data.copy()
    X=pyten.tenclass.Tensor(X)
    ##Setting Parameters
    # Construct Omega if no input
    if Omega is None:
        Omega=X.data*0+1

    # Extract number of dimensions and norm of X.
    N = X.ndims
    dimorder=range(N)  #'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}


    # Define convergence tolerance & maximum iteration
    fitchangetol = 1e-4
    maxiters = maxiter

    # Define size for factorization matrices
    if type(R) == int:
        R = R * np.ones(N, dtype=int)

    ## Error checking
    # Error checking on maxiters
    if maxiters < 0:
        raise ValueError('OPTS.maxiters must be positive')


    # Set up and error checking on initial guess for U.
    if type(init)==list:
        Uinit = init[:]
        if len(Uinit) != N:
            raise IndexError('OPTS.init does not have %d lists', N)
        for n in dimorder[1:]:
            if Uinit[n].shape!=(X.shape[n],R[n]):
                raise IndexError('OPTS.init{%d} is the wrong size', n)
    else:
        # Observe that we don't need to calculate an initial guess for the
        # first index in dimorder because that will be solved for in the first
        # inner iteration
        if init=='random':
            Uinit = range(N);Uinit[0]=[]
            for n in dimorder[1:]:
                Uinit[n] = np.random.random([X.shape[n],R[n]])
        elif init=='nvecs' or init=='eigs':
            # Compute an orthonormal basis for the dominant
            # Rn-dimensional left singular subspace of
            # X_(n) (0 <= n <= N-1).
            Uinit = range(N);Uinit[0]=[]
            for n in dimorder[1:]:
                print('  Computing %d leading e-vectors for factor %d.\n',R,n)
                Uinit[n] = X.nvecs(n,R); #first R leading eigenvecters
        else:
            raise TypeError('The selected initialization method is not supported')

    ## Set up for iterations - initializing U and the fit.
    U = Uinit[:]
    fit = 0

    if printitn > 0:
        print('\nTucker Alternating Least-Squares:\n')

    # Set up loop2 for recovery. If loop2=1, then we need to recover the Tensor.
    Loop2=0
    if  Omega.any():
        Loop2=1

    ## Main Loop: Iterate until convergence
    """Still need some change. Right now, we use two loops to recover a Tensor, one loop is enough."""
    normX = X.norm()
    for iter in range(1,maxiters+1):
        if Loop2==1:
            Xpast=X.data.copy()
            Xpast=pyten.tenclass.Tensor(Xpast)
            fitold=fit
        else:
            fitold=fit

        # Iterate over all N modes of the Tensor
        for n in range(N):
            tempU=U[:]
            tempU.pop(n)
            tempIndex=range(N)
            tempIndex.pop(n)
            Utilde=X
            for k in range(len(tempIndex)):
                Utilde = Utilde.ttm(tempU[k].transpose(), tempIndex[k]+1)

            # Maximize norm(Utilde x_n W') wrt W and
            # keeping orthonormality of W
            U[n] = Utilde.nvecs(n,R[n])

        # Assemble the current approximation
        core = Utilde.ttm(U[n].transpose(), n+1)

        # Construct fitted Tensor
        T1=pyten.tenclass.Ttensor(core, U)
        T=T1.totensor()

        # Compute fitting error
        if Loop2==1:
            X.data = T.data * (1-Omega) + X.data * Omega
            diff = Xpast.data-X.data
            fitchange=np.linalg.norm(diff) / normX

            normXtemp=X.norm()
            normresidual = np.sqrt( abs(normXtemp**2 - core.norm()**2))
            fit = 1 - (normresidual / normXtemp) #fraction explained by model
            fitchange = max(abs(fitold - fit),fitchange)

        else:
            normresidual = np.sqrt( abs(normX**2 - core.norm()**2))
            fit = 1 - (normresidual / normX) #fraction explained by model
            fitchange = abs(fitold - fit)


        # Print inner loop fitting change
        if printitn!=0 and iter%printitn==0:
            print ' Tucker_ALS: iterations={0}, fit = {1}, fit-delta = {2}\n'.format(iter, fit, fitchange)
            #print ' Iter ',iter,': fit = ',fit,'fitdelta = ',fitchange,'\n'
        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            break

    return T1,X
    #T: Decompose result.(Ttensor)
    #X: Recovered Tensor.






