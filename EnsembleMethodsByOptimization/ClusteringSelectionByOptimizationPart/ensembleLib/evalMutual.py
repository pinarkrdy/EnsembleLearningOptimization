import numpy as np
from operator import itemgetter
import scipy.io


 # DESCRIPTION
 #   computes mutual information (transinformation) between
 #   category labels trueclass and cl
 #   normalized by geometric mean of entropies
 #   ignores x and sfct - just for compatibility
 # output : mutual information (transinformation) between trueclass and cl

def evalMutual(trueclass,cl):
  rowLen = 1
  colLen = cl.size
  remappedcl = np.zeros((1, colLen))
  #A=np.zeros((max(cl),2+max(trueclass)))
  A=[]
  aaa=max(cl)
  for i in range(1,(max(cl)+1)):
    activepoints = [a for a, s in enumerate(cl) if  s==i]
    ccc=list(range(1, max(trueclass)+1))
    bbb=[trueclass[m] for m in activepoints]
    composition = np.zeros(len(ccc))
    for k in range(len(ccc)):
      composition[k]=bbb.count(ccc[k])
    j = 0
    j = [a for a, s in enumerate(composition) if  s==max(composition)]
    j=j[0]

    xxxx = np.concatenate(([j], [i], composition))
    A.append(xxxx)

  b = np.asarray(A)
  A=np.asarray(sorted(b, key=itemgetter(0)))
  A=np.delete(A, np.s_[0:2], axis=1)
  A_sum=np.sum(np.sum(A))

  pdf=np.divide(A,A_sum)
    #bitmedi
  px=np.sum(pdf, axis=0)
  py=np.transpose(np.sum(pdf, axis=1))
  uuu=py[np.newaxis]
  pyT=uuu.T
  hxbk=0
  hybg=0
  px[px == 0] = 1
  if len(px) > 1:
    d=np.log2(px)
    where_are_ = np.isnan(d)
    hxbk =-np.sum(np.multiply(px,d))
  if len(py) > 1:
    hybg =-np.sum(np.multiply(py,np.log2(py)))

  p=1
  if(len(py) * len(px) == 1) or (hxbk == 0) | (hybg == 0):
    p = 0;
  if p==1 :
    VAL=np.multiply(pdf, np.log2(np.divide(pdf, (pyT * px))))
    VAL = np.nan_to_num(VAL)
    p=sum(sum(VAL))/ np.sqrt(hxbk * hybg)

  return p

# trueclass = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\Y.mat')
# cl = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\cl_full.mat')
# cl=cl['cl_full'][0]
# trueclass=trueclass['Y']
# evalMutual(trueclass,np.transpose(cl))
