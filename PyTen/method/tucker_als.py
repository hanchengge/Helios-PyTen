__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import numpy as np
from PyTen.tools import tools
from PyTen.tenclass import ttensor
from PyTen.tenclass import tensor


def tucker_als(Y,R=20,Omega=None,tol=1e-4,maxiter=100,init='random',printitn=1):
    """ TUCKER_ALS Higher-order orthogonal iteration. """
    #Y: Original Tensor
    #R: The rank of the tensor you want to use for  approximation (recover or decompose).{Default: None}
    #tol: Tolerance on difference in fit.(Convergence tolerance for both cp(als) or tucker(als).){Default: 1.0e-4}
    #maxiter: Maximum number of iterations {Default: 50}
    #init: Initial guess 'random'|'nvecs'|'eigs'. {Default 'random'}
    #printitn: Print fit every n iterations; 0 for no printing.

    X=Y.data.copy();
    X=tensor.tensor(X);
    ##Setting Parameters
    # Construct Omega if no input
    if Omega==None:
        Omega=X.data*0+1;

    # Extract number of dimensions and norm of X.
    N = X.ndims;
    dimorder=range(N);  #'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}


    # Define convergence tolerance & maximum iteration
    fitchangetol = 1e-4;
    maxiters = maxiter;

    # Define size for factorization matrices
    if type(R) == int:
        R = R * np.ones([N,1]);

    ## Error checking
    # Error checking on maxiters
    if maxiters < 0:
        raise ValueError('OPTS.maxiters must be positive');


    # Set up and error checking on initial guess for U.
    if type(init)==list:
        Uinit = init[:];
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
            Uinit = range(N);Uinit[0]=[];
            for n in dimorder[1:]:
                Uinit[n] = np.random.random([X.shape[n],R[n]]);
        elif init=='nvecs' or init=='eigs':
            # Compute an orthonormal basis for the dominant
            # Rn-dimensional left singular subspace of
            # X_(n) (0 <= n <= N-1).
            Uinit = range(N);Uinit[0]=[];
            for n in dimorder[1:]:
                print('  Computing %d leading e-vectors for factor %d.\n',R,n);
                Uinit[n] = X.nvecs(n,R); #first R leading eigenvecters
        else:
            raise TypeError('The selected initialization method is not supported');

    ## Set up for iterations - initializing U and the fit.
    U = Uinit[:];
    fit = 0;

    if printitn > 0:
        print('\nTucker Alternating Least-Squares:\n');

    # Set up loop2 for recovery. If loop2=1, then we need to recover the tensor.
    Loop2=0;
    if  Omega.all():
        Loop2=1;

    ## Main Loop: Iterate until convergence
    """Still need some change. Right now, we use two loops to recover a tensor, one loop is enough."""
    for iter2 in range(1,maxiters+1):
        oldMissnorm=np.linalg.norm(X.data[Omega==0])
        normX = X.norm();
        for iter in range(1,maxiters+1):
            fitold = fit;

            # Iterate over all N modes of the tensor
            for n in range(N):
                tempU=U[:];
                tempU.pop(n);
                tempIndex=range(N);
                tempIndex.pop(n);
                Utilde=X;
                for k in range(len(tempIndex)):
                    Utilde = Utilde.ttm(tempU[k].transpose(), tempIndex[k]+1);

                # Maximize norm(Utilde x_n W') wrt W and
                # keeping orthonormality of W
                U[n] = Utilde.nvecs(n,R[n]);

           #     core = Utilde.ttm(U[n].transpose(), n+1);
           #     T=ttensor.ttensor(core, U);
           #     T=T.totensor();
           #     X.data[Omega==0]=T.data[Omega==0];

            # Assemble the current approximation
            # core = Utilde.ttm(U, n, 't');
            core = Utilde.ttm(U[n].transpose(), n+1);


            #T=ttensor.ttensor(core, U);
            #T=T.totensor();
            #X.data[Omega==0]=T.data[Omega==0];

            # Compute fit
            normresidual = np.sqrt( normX**2 - core.norm()**2 );
            fit = 1 - (normresidual / normX); #fraction explained by model
            fitchange = abs(fitold - fit);

            #normresidual = np.sqrt(abs(normX**2 - X.norm()**2));
            #fit = 1 - (normresidual / normX); #fraction explained by model
            #fitchange = abs(fitold - fit);

            # Print inner loop fitting change
            if printitn!=0 and iter%printitn==0:
                print ' Iter ',iter,': fit = ',fit,'fitdelta = ',fitchange,'\n';
            # Check for convergence
            if (iter > 1) and (fitchange < fitchangetol):
                break;

        # Construct fitted tensor
        T=ttensor.ttensor(core, U);

        # Check if we need outer loop to recover the tensor
        if  Loop2:
            break;

        # Recover the tensor
        temp=T.totensor();
        X.data[Omega==0]=temp.data[Omega==0];
        fitchange2=np.linalg.norm(X.data[Omega==0])-oldMissnorm;


        # Outer Loop fitting change
        if printitn!=0 and iter2%printitn==0:
                print ' Iter ',iter2,': fitdelta = ',fitchange2,'\n';
        if (iter2 > 1) and(fitchange2)<fitchangetol:
            break;

    return T,X;
    #T: Decompose result.(ttensor)
    #X: Recovered tensor.






