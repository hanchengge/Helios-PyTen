__author__ = 'Song'

from PyTen import Helios,create
from PyTen.tools import TenError
from PyTen.method import tucker_als,cp_als,CMTF,TNCP,AirCP


# /Users/Song/Desktop/PhD 1st Year/HELIOS_local/Python Tensor/PyTen-QQ/PyTen/test/1.csv
# /Users/Song/Desktop/PhD 1st Year/HELIOS_local/Python Tensor/PyTen-QQ/PyTen/test/12.csv
# /Users/Song/Desktop/PhD 1st Year/HELIOS_local/Python Tensor/PyTen-QQ/PyTen/test/Random.csv
# /Users/Song/Desktop/PhD 1st Year/HELIOS_local/Python Tensor/PyTen-QQ/PyTen/test/aux_CM.csv
# /Users/Song/Desktop/PhD 1st Year/HELIOS_local/Python Tensor/PyTen-QQ/PyTen/test/aux_1.csv
# /Users/Song/Desktop/PhD 1st Year/HELIOS_local/Python Tensor/PyTen-QQ/PyTen/test/aux_2.csv
# /Users/Song/Desktop/PhD 1st Year/HELIOS_local/Python Tensor/PyTen-QQ/PyTen/test/aux_3.csv
#  /Users/Song/Desktop/PhD 1st Year/HELIOS_local/Python Tensor/PyTen-QQ/a.npy

[OriTensor,DeTensor,TenClass,RecTensor]=Helios.Helios();
#[Origin,FinalTensor,FinalTucker]=Helios.Helios();
#[Origin,FinalTensor,FinalTucker]=Helios.Helios(FunctionName='1',);

#[Origin,FinalTensor,FinalTucker]=Helios.Helios();
#[Origin,FinalTensor,FinalTucker]=Helios.Helios();
#[Origin,FinalTensor,FinalTucker]=Helios.Helios(FileName='/Users/Song/Desktop/1.csv',FunctionName='cp_als',R=2,tol=1e-4,maxiter=50,init='random',printitn=1);

print DeTensor.data
print OriTensor
print TenClass
print RecTensor.data

from PyTen import Helios,create
from PyTen.tools import TenError
from PyTen.method import tucker_als,cp_als,CMTF,TNCP,AirCP
import numpy as np

siz=[100,100,100];
R=10;
M=0.5;
tp='Tucker';

#[X,Omega,sol]=create.create(siz,R,M,tp);
#[T,rX]=tucker_als.tucker_als(X,R);
#[T,rX]=tucker_als.tucker_als(X,R,Omega,printitn=0);
#[T,rX]=tucker_als.tucker_als(X,R,Omega,printitn=0);


[X1,Omega1,sol1]=create.create(siz,R,M,tp);
[T1,rX1]=cp_als.cp_als(X1,R);
[T1,rX1]=cp_als.cp_als(X1,R,Omega1,maxiter=100);
[T2,rX2]=tucker_als.tucker_als(X1,R,Omega1,maxiter=100,printitn=0);

#sqq=sol.totensor().data-rX1.data;
np.linalg.norm(sol1.totensor().tondarray()-rX1.data)
np.linalg.norm(sol1.totensor().tondarray()-rX1.data)/np.linalg.norm(sol1.totensor().tondarray())

[Err,ReErr1,ReErr2]=TenError.TenError(sol1.totensor(),rX1,Omega1)


np.linalg.norm(sol1.totensor().tondarray()-rX2.data)
np.linalg.norm(sol1.totensor().tondarray()-rX2.data)/np.linalg.norm(sol1.totensor().tondarray())




self=AirCP.AirCP(X1,Omega1,rank=2);
self1=TNCP.TNCP(X1,Omega1,rank=2);
#self.initializeLatentMatrices()
self.run()
np.linalg.norm(sol1.totensor().tondarray()-self.X.data)
np.linalg.norm(sol1.totensor().tondarray()-self.X.data)/np.linalg.norm(sol1.totensor().tondarray())
self1.run()
np.linalg.norm(sol1.totensor().tondarray()-self1.X.data)
np.linalg.norm(sol1.totensor().tondarray()-self1.X.data)/np.linalg.norm(sol1.totensor().tondarray())

print sol1.totensor().data
print rX1.data;



sqq=sol1.us[0];
V0=np.random.random([10,50]);
Y=np.dot(sqq,V0);
[T,Rec,V]=CMTF.CMTF(X1,Y,1,R,Omega1,maxiter=500,printitn=10);
CM1=np.dot(T.Us[0],V.T);
np.linalg.norm(sol1.totensor().tondarray()-Rec.data)
np.linalg.norm(sol1.totensor().tondarray()-Rec.data)/np.linalg.norm(sol1.totensor().tondarray())

np.linalg.norm(CM1-Y)
np.linalg.norm(CM1-Y)/np.linalg.norm(Y)









import numpy
from PyTen.tenclass import tensor
x=numpy.array([[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]]])
x=tensor.tensor(x)
[T,UsInit]=cp_als.cp_als(x,2);


x1=numpy.array([[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]]])
x1[1,1,1]=0;
Omega=x1*0+1;
Omega[1,1,1]=0;
x1=tensor.tensor(x1)
[T1,UsInit1]=cp_als.cp_als(x1,2,Omega);
[Origin,FinalTensor,FinalTucker]=Helios.Helios(FileName='/Users/Song/Desktop/1.csv',FunctionName='cp_als',
                                               Recover='1',Omega=1,R=2,tol=1e-4,maxiter=50,init='random',printitn=1);




from PyTen import Helios,create
from PyTen.method import tucker_als,cp_als,CMTF,AirCP
import numpy
from PyTen.tenclass import tensor
x=numpy.array([[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]]])
x=tensor.tensor(x)
[T,UsInit]=cp_als.cp_als(x,2);

y=numpy.array([[1, 4],[2, 5], [3, 6]]);

[T,Rec,V]=CMTF.CMTF(x,y,1,2);
numpy.dot(T.Us[0],V.T)



from PyTen.method import tucker_als,cp_als,CMTF,AirCP,TNCP
import numpy as np
from PyTen.tenclass import tensor
x1=np.array([[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]]])
x2=x1.copy()
x1[1,1,1]=0
Omega=x1*0+1
Omega[1,1,1]=0
Omega1=tensor.tensor(Omega)
x1=tensor.tensor(x1)
[T1,UsInit1]=cp_als.cp_als(x1,2,Omega)
self=AirCP.AirCP(x1,Omega1,rank=20)
self1=TNCP.TNCP(x1,Omega1,rank=20)
#self.initializeLatentMatrices()
self.run()
self1.run()
full = self.II.copy()
for i in range(self.ndims):
    full=full.ttm(self.U[i], i+1)




from PyTen.method import onlineCP
import numpy as np
from PyTen.tenclass import tensor
initX=np.array([[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]]]);
newX1=np.array([[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]]]);
newX2=np.array([[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]]]);

initX=tensor.tensor(initX);
newX1=tensor.tensor(newX1);
newX2=tensor.tensor(newX2);

self=onlineCP.onlineCP(initX,rank=2,tol=1e-8);
self.update(newX1)
self.update(newX2)




from PyTen.method import OLSGD
import numpy as np
from PyTen.tenclass import tensor
initX=np.array([[[1, 4],[2, 5], [3, 6]]]);
newX1=np.array([[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]]]);
newX2=np.array([[[1, 4],[2, 5], [3, 6]],[[1, 4],[2, 5], [3, 6]]]);
initX[0,0]=0
newX1[1,1]=0
newX2[1,1,0]=0;newX2[0,1,1]=0

initXO=initX*0+1;initXO[0,0]=0
newX1O=newX1*0+1;newX1O[1,1]=0
newX2O=newX2*0+1;newX2O[1,1,0]=0;newX2O[0,1,1]=0

initX=tensor.tensor(initX);
newX1=tensor.tensor(newX1);
newX2=tensor.tensor(newX2);

initXO=tensor.tensor(initXO);
newX1O=tensor.tensor(newX1O);
newX2O=tensor.tensor(newX2O);

self=OLSGD.OLSGD(rank=2, mu=0.01,lmbda=0.1);
self.update(initX,initXO,rank=2)
self.update(newX1,newX1O,2)

self.update(newX2,newX2O,2)




newX3=np.array([[0, 0],[0, 0], [0, 0]])
newX3=tensor.tensor(newX3)
