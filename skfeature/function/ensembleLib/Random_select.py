import scipy.optimize
import numpy as np
from skfeature.function.ensembleLib import clusterEnsemble,evalMutual

def Random_select(Y,c,bestSize,PRED):
    r =np.random.random_integers(1,150,bestSize)
    PRED = PRED[:, r]
    cls = PRED;
    [cl, qual] = clusterEnsemble.clusterEnsemble(cls.T,c)
    cn = Y;
    mutualInfo = evalMutual.evalMutual(cn, cl)
    return mutualInfo



# trueclass = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\Y.mat')
# pred = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\PRED.mat')
# PRED = pred['PRED']
# Y = trueclass['Y']
# c=8
# bestSize=46
# Random_select(Y,c,bestSize,PRED)
