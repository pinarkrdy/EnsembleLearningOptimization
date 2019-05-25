import numpy as np
import scipy.optimize
from skfeature.function.dccp import dccpFunc
from skfeature.function.ensembleLib import evalMutual,clusterEnsemble,objCluster
from sklearn.metrics import normalized_mutual_info_score
import scipy.optimize as op
from skfeature.function.ensembleLib import clusterEnsemble,evalMutual
import ensemble.Cluster_Ensembles as ce
import numpy as np

def Cluster_select(IDX,pred,nmi,SNMI,bestSize):
    b = np.array([IDX])
    IDX=b.T

    Ensemble_subset = []
    for i in range(0,bestSize-1):
        Group_id = np.where(IDX == i)[0]
        Group_SNMI = SNMI[Group_id]


        max_SNMI_ind = np.argmax(Group_SNMI)
        max_SNMI = Group_SNMI[max_SNMI_ind]

        position = np.where(Group_SNMI == max_SNMI)[0]
        last_id = Group_id[position]
        Ensemble_subset.append(np.array(last_id))
    flattened = [val for sublist in Ensemble_subset for val in sublist]
    return np.asarray(flattened)
#
# idx = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\IDX.mat')
# Nmi = scipy.io.loadmat('C:/MATLAB\mfilesICML/EnsembleClustering/ClusterEnsemble-V2.0/nmi.mat')
# pred = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\PRED.mat')
# SNMI_ = scipy.io.loadmat('C:/MATLAB\mfilesICML/EnsembleClustering/ClusterEnsemble-V2.0/SNMI.mat')
# PRED=pred['PRED']
# nmi=Nmi['nmi']
# IDX=idx['IDX']
# SNMI=SNMI_['SNMI'][0]
# bestSize=50
# Cluster_select(IDX,pred,nmi,SNMI,bestSize)
