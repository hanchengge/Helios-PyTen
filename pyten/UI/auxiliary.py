__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import pandas as pd
import numpy as np
import scipy

import pyten.tenclass
import pyten.method


def auxiliary(FileName=None, FunctionName=None, AuxMode=None, AuxFile=None, Recover=None, Omega=None, R=2, tol=1e-8,
              maxiter=100, init='random', printitn=0):
    """ Helios API returns decomposition or Recovery with Auxiliary Result
    arg can be list, tuple, set, and array with numerical values.
    # FileName: {Default: None}
    # FunctionName: Tensor-based Method
    # Recover: Input '1' to recover other to decompose.{Default: None}
    # R: The rank of the Tensor you want to use for  approximation (recover or decompose).{Default: 2}
    # tol: Tolerance on difference in fit.(Convergence tolerance for both cp(als) or tucker(als).){Default: 1.0e-4}
    # maxiter: Maximum number of iterations {Default: 50}
    # init: Initial guess 'random'|'nvecs'|'eigs'. {Default 'random'}
    # printitn: Print fit every n iterations; 0 for no printing."""

    # User Interface
    if FileName is None:
        FileName = raw_input("Please input the FileName of the Tensor data:\n")
        print("\n")

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

    if FunctionName is None:
        FunctionName = raw_input("Please choose the method you want to use to recover data(Input one number):\n"
                                 " 1. AirCP  2.CMTF 0.Exit \n")
        print("\n")

    if FunctionName == '1' or FunctionName == 'AirCP':
        simMats = np.array([np.identity(X.shape[i]) for i in range(X.ndims)])
        if AuxMode is None:
            AuxMode = raw_input(
                "Please input all the modes that have Auxiliary Similarity Matrix (separate with space. Input 'None' if no auxiliary info.)\n")
            if AuxMode != 'None':
                for i in range((len(AuxMode) + 1) / 2):
                    Mode = int(AuxMode[i * 2])
                    FileName2 = raw_input(
                        "Please input the FileName of the Auxiliary Similarity Matrix of Mode " + str(Mode) + " :\n")
                    if FileName2 != 'None':
                        dat2 = pd.read_csv(FileName2, delimiter=';', header=None)
                        ## Data preprocessing
                        Mat_dat = dat2.values
                        if Mat_dat.shape == (X1.shape[Mode - 1], X1.shape[Mode - 1]):
                            simMats[Mode - 1] = Mat_dat
                        else:
                            print('Wrong Size of Auxiliary Matrix.\n')
                print("\n")
        else:
            for i in range((len(AuxMode) + 1) / 2):
                Mode = int(AuxMode[i * 2])
                FileName2 = AuxFile[i]
                if FileName2 != 'None':
                    dat2 = pd.read_csv(FileName2, delimiter=';', header=None)
                    ## Data preprocessing
                    Mat_dat = dat2.values
                    if Mat_dat.shape == (X1.shape[Mode - 1], X1.shape[Mode - 1]):
                        simMats[Mode - 1] = Mat_dat
                    else:
                        print('Wrong Size of Auxiliary Matrix.\n')




    elif FunctionName == '2' or FunctionName == 'CMTF':
        CM = None
        Y = None
        if AuxMode is None:
            AuxMode = raw_input(
                "Please input all the modes that have Coupled Matrix (separate with space. Input 'None' if no coupled matrices. Allow Multiple Coupled Matrices for One Mode)\n")
            if AuxMode != 'None':
                for i in range((len(AuxMode) + 1) / 2):
                    Mode = int(AuxMode[i * 2])
                    FileName2 = raw_input("Please input the FileName of the Coupled Matrix of Mode " + str(
                        Mode) + " (Input 'None' if no auxiliary info):\n")
                    print("\n")
                    if FileName2 != 'None':
                        dat2 = pd.read_csv(FileName2, delimiter=';')
                        Mat_dat = dat2.values
                        Mat_subs = Mat_dat[:, range(2)]
                        Mat_subs = Mat_subs - 1
                        Mat_vals = Mat_dat[:, 2]
                        Mat_siz = np.max(Mat_subs, 0)
                        Mat_siz = Mat_siz + 1
                        X2 = scipy.sparse.coo_matrix((Mat_vals, (Mat_subs[:, 0], Mat_subs[:, 1])),
                                                     shape=(Mat_siz[0], Mat_siz[1]))
                        if CM is None:
                            CM = Mode
                            Y = X2.toarray()
                        else:
                            CM = [CM, Mode]
                            Y = [Y, X2.toarray()]
        else:
            for i in range((len(AuxMode) + 1) / 2):
                Mode = int(AuxMode[i * 2])
                FileName2 = AuxFile[i]
                print("\n")
                if FileName2 != 'None':
                    dat2 = pd.read_csv(FileName2, delimiter=';')
                    Mat_dat = dat2.values
                    Mat_subs = Mat_dat[:, range(2)]
                    Mat_subs = Mat_subs - 1
                    Mat_vals = Mat_dat[:, 2]
                    Mat_siz = np.max(Mat_subs, 0)
                    Mat_siz = Mat_siz + 1
                    X2 = scipy.sparse.coo_matrix((Mat_vals, (Mat_subs[:, 0], Mat_subs[:, 1])),
                                                 shape=(Mat_siz[0], Mat_siz[1]))
                    if CM is None:
                        CM = Mode
                        Y = X2.toarray()
                    else:
                        CM = [CM, Mode]
                        Y = [Y, X2.toarray()]


    elif FunctionName == '0':
        print 'Successfully Exit'
        return None, None, None, None
    else:
        raise ValueError('No Such Method')

    if Recover is None:
        Recover = raw_input("If there are missing values in the file? (Input one number)\n"
                            "1. Yes, recover it  2.No, just decompose(Nonexistent number will be replaced by 0) 0.Exit\n")


    # Construct Omega
    output = 1  # An output indicate flag. (Recover:1, Decompose: 0)
    if type(Omega) != np.ndarray:
        Omega = X.data * 0 + 1
        if Recover == '1':
            Omega[lstnan] = 0
            output = 2

    # Choose method to recover or decompose
    if type(FunctionName) == str:
        if FunctionName == '1' or FunctionName == 'AirCP':
            Omega1 = pyten.tenclass.Tensor(Omega)
            self = pyten.method.AirCP(X, Omega1, R, tol, maxiter, simMats=simMats)
            self.run()
            Final = self.U
            Rec = self.X
            full = self.II.copy()
            for i in range(self.ndims):
                full = full.ttm(self.U[i], i + 1)

        elif FunctionName == '2' or FunctionName == 'CMTF':
            [Final, Rec, V] = pyten.method.cmtf(X, Y, CM, R, Omega, tol, maxiter, init, printitn)
            full = Final.totensor()

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
        print "\n" + "The original Tensor is: "
        print Ori
        print "\n" + "The Decomposed Result is: "
        print Final
    else:
        newsubs = Rec.tosptensor().subs
        tempvals = Rec.tosptensor().vals
        newfilename = FileName[:-4] + '_Recover' + FileName[-4:]
        print  "\n" + ("The original Tensor is: ")
        print  Ori
        print  "\n" + ("The Recovered Tensor is: ")
        print  Rec.data


    # Reconstruct
    df = dat1
    for i in range(nv):
        pos = map(sum, newsubs == subs[i])
        idx = pos.index(nd)
        temp = tempvals[idx]
        df.iloc[i, nd] = temp[0]
        # newvals.append(list(tempvals(idx)))
    df.to_csv(newfilename, sep=';', index=0)


    # Return result
    return Ori, full, Final, Rec
    # Ori:   Original Tensor
    # full:  Full Tensor reconstructed by decomposed matrices
    # Final: Decomposition Results e.g. Ttensor or Ktensor
    # Rec:   Recovered Tensor (Completed Tensor)
