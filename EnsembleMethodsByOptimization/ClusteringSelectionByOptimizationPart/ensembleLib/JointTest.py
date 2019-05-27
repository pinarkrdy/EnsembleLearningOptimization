import numpy as np
from EnsembleMethodsByOptimization.ClusteringSelectionByOptimizationPart.ensembleLib import clusterEnsemble
import scipy.optimize as op
from EnsembleMethodsByOptimization.ClusteringSelectionByOptimizationPart.ensembleLib import clusterEnsemble,evalMutual
# This test basically does the following
#     load the true Labels Y
#     load PRED matrix which is the pool
#     perform cluster ensemble algorithm
# Parameters
#     ----------
#     Y : int
#     PRED=array of shape - rediction result
#     Subset_joint :array of shape
#     C : int
#     Returns
#     -------
#     MutualInfo mutual info score on prediction matrix
#
def JointTest(Y,PRED,Subset_joint,c):
    #asagidaki satiri sonra remove et
    #Subset_joint[0]=Subset_joint[0]-1
    Pruned = PRED[:,Subset_joint]
    cn=Y
    cls = Pruned
    cl, qual = clusterEnsemble.clusterEnsemble(np.transpose(cls), c)
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
