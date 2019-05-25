from skfeature.function.ensembleLib import clusterEnsemble,evalMutual
import numpy as np
import scipy.optimize
def ClusterSelectTest(Y,PRED,Ensemble_subset,c):
    # Subset = [val for sublist in Ensemble_subset for val in sublist]
    # Subset=np.asarray(Subset)

    Pruned = PRED[:, Ensemble_subset]
    cn = Y
    cls = Pruned
    cl,qual = clusterEnsemble.clusterEnsemble(cls.T,c)
    MutualInfo = evalMutual.evalMutual(cn, cl)
    return MutualInfo


# pred = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\PRED.mat')
# trueclass = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\Y.mat')
# Ensemblesubset = scipy.io.loadmat('C:/MATLAB\mfilesICML/EnsembleClustering/ClusterEnsemble-V2.0/Ensemble_subset.mat')
# PRED=pred['PRED']
# Y=trueclass['Y']
# Ensemble_subset=Ensemblesubset['Ensemble_subset']
# c=8
#
# ClusterSelectTest(Y,PRED,Ensemble_subset,c)
