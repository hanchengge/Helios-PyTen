__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import pandas as pd
import numpy as np

import pyten.tenclass
import pyten.method


def dynamic(FileName=None,FunctionName=None,Fore_File=None,Save_File=None,Recover=None,Omega=None,R=2,tol=1e-8,maxiter=100,init='random', printitn=0, mu=0.01,lmbda=0.1,self=None):
    """ Helios UI for dynamic scenario returns Online decomposition or Recovery Result
    arg can be list, tuple, set, and array with numerical values. """
    #FileName: {Default: None}
    #FunctionName: Tensor-based Method
    #Recover: Input '1' to recover other to decompose.{Default: None}
    #R: The rank of the Tensor you want to use for  approximation (recover or decompose).{Default: 2}
    #tol: Tolerance on difference in fit.(Convergence tolerance for both cp(als) or tucker(als).){Default: 1.0e-4}
    #maxiter: Maximum number of iterations {Default: 50}
    #init: Initial guess 'random'|'nvecs'|'eigs'. {Default 'random'}
    #printitn: Print fit every n iterations; 0 for no printing.
    #User Interface

    if FileName==None:
        FileName = raw_input("Please input the FileName of the data:\n")
        print("\n")

    if FunctionName==None:
        FunctionName = raw_input("Please choose the method you want to use (Input one number):\n"
                                 " 1. onlineCP(only for decomposition)  2.OLSGD 0.Exit \n")
        print("\n")

    Former_result='2'
    if self==None and Fore_File==None:
        Former_result=raw_input("If there are former decomposition or recovery result (.npy file):\n"
                                 " 1. Yes  2.No 0.Exit \n")
        if Former_result=='1':
            Fore_File = raw_input("Please input the FileName of the former result:\n")
            temp=np.load(Fore_File)
            self=temp.all()
    elif self==None:
        Former_result='1'
        temp=np.load(Fore_File)
        self=temp.all()

    if Recover==None:
        if FunctionName=='1':
            Recover = '2'
        else:
            Recover = raw_input("If there are missing values in the file? (Input one number)\n"
                            "1. Yes, recover it  2.No, just decompose(Nonexistent number will be replaced by 0) 0.Exit\n")

    # Use pandas package to load data
    dat1=pd.read_csv(FileName,delimiter=';')

    ## Data preprocessing
    #First: create Sptensor
    dat=dat1.values
    sha=dat.shape
    subs=dat[:,range(sha[1]-1)]
    subs=subs-1
    vals=dat[:,sha[1]-1]
    vals=vals.reshape(len(vals),1)
    siz = np.max(subs,0)
    siz = siz+1;
    X1=pyten.tenclass.Sptensor(subs, vals,siz)

    #Second: create Tensor object and find missing data
    X=X1.totensor()
    Ori=X.data
    lstnan = np.isnan(X.data)
    X.data=np.nan_to_num(X.data)

    #Construct Omega
    output=1; #An output indicate flag. (Recover:1, Decompose: 0)
    if type(Omega)!=np.ndarray:
        Omega=X.data*0+1
        if Recover=='1':
            Omega[lstnan]=0
            output=2

    #Choose method to recover or decompose
    if type(FunctionName)==str:
        n=X.shape[0]
        if  FunctionName=='1' or FunctionName=='onlineCP':
            if Former_result=='1':
                self.update(X)
            else:
                self=pyten.method.onlineCP(X,R,tol)
            Final=self.As
            full=pyten.tenclass.Tensor(self.X.data[-n:,])
            Rec=None

        elif FunctionName=='2' or FunctionName=='OLSGD':
            Omega1=pyten.tenclass.Tensor(Omega)
            if Former_result!='1':
                self=pyten.method.OLSGD(R, mu, lmbda)
            self.update(X, Omega1, R, mu,lmbda)
            Final=[self.A, self.B, self.C]
            full=pyten.tenclass.Tensor(self.X.data[-n:,:,:])
            Rec=pyten.tenclass.Tensor(self.Rec.data[-n:,:,:])
        elif FunctionName=='0':
            return 'Successfully Exit'
        else:
            raise ValueError('No Such Method')

    else:
        raise TypeError('No Such Method')


    #Output Result
    [nv,nd]=subs.shape
    if  output==1:
        newsubs=full.tosptensor().subs
        tempvals=full.tosptensor().vals
        newfilename=FileName[:-4]+'_Decomposite'+FileName[-4:]
        print  "\n"+("The original Tensor is: "); print  Ori
        print  "\n"+("The Decomposed Result is: "); print  Final
    else:
        newsubs=Rec.tosptensor().subs
        tempvals=Rec.tosptensor().vals
        newfilename=FileName[:-4]+'_Recover'+FileName[-4:]
        print  "\n"+("The original Tensor is: "); print  Ori
        print  "\n"+("The Recovered Tensor is: "); print  Rec.data


    #Reconstruct
    df=dat1;
    for i in range(nv):
        pos=map(sum,newsubs==subs[i])
        idx=pos.index(nd)
        temp=tempvals[idx]
        df.iloc[i,nd]=temp[0]
        #newvals.append(list(tempvals(idx)));
    df.to_csv(newfilename,sep=';',index=0)

    if Save_File==None:
        SaveOption=raw_input("If you want to save the result into .npy file):\n"
                                     " 1. Yes  2.No  0.Exit \n")
        if SaveOption=='1':
            Save_File = raw_input("Please input the address and fileName to save the result: (end in '.npy')\n")
            np.save(Save_File,self)

    #Return result
    return Ori,full,Final,Rec
    # Ori:   Original Tensor
    # full:  Full Tensor reconstructed by decomposed matrices
    # Final: Decomposition Results e.g. Ttensor or Ktensor
    # Rec:   Recovered Tensor (Completed Tensor)

