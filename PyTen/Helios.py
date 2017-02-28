__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

from PyTen import Helios1,Helios2,Helios3


def Helios(Scenario=None):
    """ Helios Main API returns decomposition or Recovery Result of All three Scenarios """
    #FileName=None,FunctionName=None,Recover=None,Omega=None,R=2,tol=1e-8,maxiter=100,init='random',printitn=0
    #FileName: {Default: None}
    #FunctionName: Tensor-based Method
    #Recover: Input '1' to recover other to decompose.{Default: None}
    #R: The rank of the tensor you want to use for  approximation (recover or decompose).{Default: 2}
    #tol: Tolerance on difference in fit.(Convergence tolerance for both cp(als) or tucker(als).){Default: 1.0e-4}
    #maxiter: Maximum number of iterations {Default: 50}
    #init: Initial guess 'random'|'nvecs'|'eigs'. {Default 'random'}
    #printitn: Print fit every n iterations; 0 for no printing.

    # User Interface
    if Scenario==None:
        Scenario = raw_input("Please choose the Scenario:\n"
                                 " 1. Basic Tensor Decomposition/Completion  2.Tensor Completion with Auxiliary Information 3.Dynamic Tensor Decomposition 0.Exit \n");
        print("\n");

    if Scenario=='1': #Basic Tensor Decomposition/Completion
        [Ori,full,Final,Rec]=Helios1.Helios1()
    elif Scenario=='2': #Tensor Completion with Auxiliary Information
        [Ori,full,Final,Rec]=Helios2.Helios2()
    elif Scenario=='3': #Dynamic Tensor Decomposition
        [Ori,full,Final,Rec]=Helios3.Helios3()
    elif Scenario=='0':
        return 'Successfully Exit'
    else:
        raise ValueError('No Such Scenario')


    #Return result
    return Ori,full,Final,Rec
    #Ori: Original Tensor
    #full: Decomposed tensor.
    #Final: ttensor or ktensor.
    #Rec: Recovered tensor.


