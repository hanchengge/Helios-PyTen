__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import pandas as pd
import numpy as np

import pyten.tenclass
import pyten.method


def basic(FileName=None, FunctionName=None, Recover=None, Omega=None, R=2, tol=1e-8, maxiter=100, init='random',
          printitn=0):
    """ Helios1 API returns CP_ALS, TUCKER_ALS, or NNCP decomposition or Recovery Result
    arg can be list, tuple, set, and array with numerical values.
    #FileName: {Default: None}
    #FunctionName: Tensor-based Method
    #Recover: Input '1' to recover other to decompose.{Default: None}
    #R: The rank of the Tensor you want to use for  approximation (recover or decompose).{Default: 2}
    #tol: Tolerance on difference in fit.(Convergence tolerance for both cp(als) or tucker(als).){Default: 1.0e-4}
    #maxiter: Maximum number of iterations {Default: 50}
    #init: Initial guess 'random'|'nvecs'|'eigs'. {Default 'random'}
    #printitn: Print fit every n iterations; 0 for no printing."""

    # User Interface
    if FileName is None:
        FileName = raw_input("Please input the FileName of the data:\n")
        print("\n")

    if FunctionName is None:
        FunctionName = raw_input("Please choose the method you want to use to recover data(Input one number):\n"
                                 " 1. Tucker(ALS)  2.CP(ALS) 3.NNCP(Trace Norm + ADMM, Only For Recovery) "
                                 "4.SiLRTC(Only For Recovery) 5.FaLRTC(Only For Recovery) 6.HaLRTC(Only For Recovery)  0.Exit \n")
        print("\n")
    if Recover is None:
        Recover = raw_input("If there are missing values in the file? (Input one number)\n"
                            "1. Yes, recover it  2.No, just decompose(Nonexistent number will be replaced by 0) 0.Exit\n")


    # Use pandas package to load data
    dat1 = pd.read_csv(FileName, delimiter=';')

    ## Data preprocessing
    # First: create Sptensor
    dat = dat1.values
    sha = dat.shape
    subs = dat[:, range(sha[1] - 1)]
    subs = subs - 1
    vals = dat[:, sha[1] - 1]
    vals = vals.reshape(len(vals), 1)
    siz = np.max(subs, 0)
    siz = np.int32(siz + 1)
    X1 = pyten.tenclass.Sptensor(subs, vals, siz)

    # Second: create Tensor object and find missing data
    X = X1.totensor()
    Ori = X.data
    lstnan = np.isnan(X.data)
    X.data = np.nan_to_num(X.data)

    # Construct Omega
    output = 1  # An output indicate flag. (Recover:1, Decompose: 0)
    if type(Omega) != np.ndarray:
        # if True in lstnan:
        Omega = X.data * 0 + 1
        Omega[lstnan] = 0
        if Recover == '1':
            output = 2

    # Choose method to recover or decompose
    if type(FunctionName) == str:
        if FunctionName == '1' or FunctionName == 'tucker_als':
            [Final, Rec] = pyten.method.tucker_als(X, R, Omega, tol, maxiter, init, printitn)
            full = Final.totensor()
        elif FunctionName == '2' or FunctionName == 'cp_als':
            [Final, Rec] = pyten.method.cp_als(X, R, Omega, tol, maxiter, init, printitn)
            full = Final.totensor()
        elif FunctionName == '3' or FunctionName == 'NNCP':
            Omega1 = pyten.tenclass.Tensor(Omega)
            NNCP = pyten.method.TNCP(X, Omega1, R, tol, maxiter)
            NNCP.run()
            Final = NNCP.U
            Rec = NNCP.X
            full = NNCP.II.copy()
            for i in range(NNCP.ndims):
                full = full.ttm(NNCP.U[i], i + 1)
        elif FunctionName == '4' or FunctionName == 'SiLRTC':
            Rec = pyten.method.silrtc(X, Omega, maxIter=maxiter, printitn=printitn)
            full = None
            Final = None
        elif FunctionName == '5' or FunctionName == 'FaLRTC':
            Rec = pyten.method.falrtc(X, Omega, maxIter=maxiter, printitn=printitn)
            full = None
            Final = None
        elif FunctionName == '6' or FunctionName == 'HaLRTC':
            Rec = pyten.method.halrtc(X, Omega, maxIter=maxiter, printitn=printitn)
            full = None
            Final = None
        elif FunctionName == '0':
            print 'Successfully Exit'
            return None, None, None, None
        else:
            raise ValueError('No Such Method')

    else:
        raise TypeError('No Such Method')


    # Output Result
    [nv, nd] = subs.shape
    if output == 1:
        newsubs = full.tosptensor().subs
        tempvals = full.tosptensor().vals
        newfilename = FileName[:-4] + '_Decomposite' + FileName[-4:]
        print  "\n" + ("The original Tensor is: ");
        print  Ori
        print  "\n" + ("The Decomposed Result is: ");
        print  Final
    else:
        newsubs = Rec.tosptensor().subs
        tempvals = Rec.tosptensor().vals
        newfilename = FileName[:-4] + '_Recover' + FileName[-4:]
        print  "\n" + ("The original Tensor is: ");
        print  Ori
        print  "\n" + ("The Recovered Tensor is: ");
        print  Rec.data


    # Reconstruct
    df = dat1
    for i in range(nv):
        pos = map(sum, newsubs == subs[i])
        idx = pos.index(nd)
        temp = tempvals[idx]
        df.iloc[i, nd] = temp[0]
        # newvals.append(list(tempvals(idx)));
    df.to_csv(newfilename, sep=';', index=0)


    # Return result
    return Ori, full, Final, Rec
    # Ori:   Original Tensor
    # full:  Full Tensor reconstructed by decomposed matrices
    # Final: Decomposition Results e.g. Ttensor or Ktensor
    # Rec:   Recovered Tensor (Completed Tensor)
