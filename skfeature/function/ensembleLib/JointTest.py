import numpy as np
import scipy.optimize
from skfeature.function.dccp import dccpFunc
from skfeature.function.ensembleLib import evalMutual,clusterEnsemble,objCluster
from sklearn.metrics import normalized_mutual_info_score
import scipy.optimize as op
from skfeature.function.ensembleLib import clusterEnsemble,evalMutual
def JointTest(Y,PRED,Subset_joint,c):
    #asagidaki satiri sonra remove et
    #Subset_joint[0]=Subset_joint[0]-1
    Pruned = PRED[:,Subset_joint]
    cn=Y
    cls = Pruned
    cl, qual = clusterEnsemble.clusterEnsemble(np.transpose(cls),c)
    MutualInfo = evalMutual.evalMutual(cn, cl)
    return MutualInfo


# trueclass = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\Y.mat')
# Subset_joint = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\Subset_joint.mat')
# pred = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\PRED.mat')
# PRED=pred['PRED']
# Subsetjoint=Subset_joint['Subset_joint']
# Y=trueclass['Y']
# c = 8
# aaa=JointTest(Y,PRED,Subsetjoint,c)
