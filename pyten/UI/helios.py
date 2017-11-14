__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import pyten.UI


def helios(Scenario=None):
    """ Helios Main API returns decomposition or Recovery Result of All three Scenarios
    #FileName=None,FunctionName=None,Recover=None,Omega=None,R=2,tol=1e-8,maxiter=100,init='random',printitn=0
    #FileName: {Default: None}
    #FunctionName: Tensor-based Method
    #Recover: Input '1' to recover other to decompose.{Default: None}
    #R: The rank of the Tensor you want to use for  approximation (recover or decompose).{Default: 2}
    #tol: Tolerance on difference in fit.(Convergence tolerance for both cp(als) or tucker(als).){Default: 1.0e-4}
    #maxiter: Maximum number of iterations {Default: 50}
    #init: Initial guess 'random'|'nvecs'|'eigs'. {Default 'random'}
    #printitn: Print fit every n iterations; 0 for no printing."""

    # Initialization
    Ori = None  # Original Tensor
    full = None  # Full Tensor reconstructed by decomposed matrices
    Final = None  # Decomposition Results e.g. Ttensor or Ktensor
    Rec = None  # Recovered Tensor (Completed Tensor)

    # User Interface
    if Scenario is None:
        Scenario = raw_input("Please choose the Scenario:\n"
                             " 1. Basic Tensor Decomposition/Completion  2.Tensor Completion with Auxiliary Information 3.Dynamic Tensor Decomposition 0.Exit \n")

    if Scenario == '1':  # Basic Tensor Decomposition/Completion
        [Ori, full, Final, Rec] = pyten.UI.basic()
    elif Scenario == '2':  # Tensor Completion with Auxiliary Information
        [Ori, full, Final, Rec] = pyten.UI.auxiliary()
    elif Scenario == '3':  # Dynamic Tensor Decomposition
        [Ori, full, Final, Rec] = pyten.UI.dynamic()
    elif Scenario == '0':
        print 'Successfully Exit'
        return Ori, full, Final, Rec
    else:
        raise ValueError('No Such Scenario')

    # Return result
    return Ori, full, Final, Rec
