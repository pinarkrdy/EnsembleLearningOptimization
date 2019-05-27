import numpy as np
import mosek
import numpy as np
from numpy import genfromtxt
import scipy.optimize
from EnsembleMethodsByOptimization.dccp import dccpFunc
from EnsembleMethodsByOptimization.ClusteringSelectionByOptimizationPart.ensembleLib import evalMutual, objCluster
from EnsembleMethodsByOptimization.ClusteringSelectionByOptimizationPart.ensembleLib import clusterEnsemble
from sklearn.metrics import normalized_mutual_info_score
import scipy.optimize as op
 # This script analyzes Joint Criterion Method
 # The method takes a simple and straight forward approach and combines
 # the quality and diversity into a joint criterion function

def joint_criterion(bestSize,PRED,T,SNMI):
    Desired_Size = int(bestSize)
    alpha = 0.5
    index_set = list(range(150))
    Subset_joint=[]
    sorted_T_val=-np.sort(-SNMI)[0]
    sorted_T_ind=np.argsort(-SNMI)[0]
    aaa =sorted_T_ind
    index_set[int(aaa)] = 0
    Subset_joint.append(aaa)
    new_subset=[]
    for k in range(Desired_Size - 1):
        for j, ind in enumerate(Subset_joint):
            index_set[ind] = 0
        # index_set[np.asarray(Subset_joint)] = 0
        M = np.nonzero(index_set)[0]
        Tnew=np.zeros((150,150))
        A = np.zeros(len(M))
        for s_ind in range(len(M)):
            if index_set[s_ind] != 0:
                new_subset.extend(Subset_joint)
                new_subset.append(index_set[s_ind])
                for p in range(len(new_subset)):
                    for k in range(len(new_subset)):
                        Tnew[p,k] = T[new_subset[p],new_subset[k]]
                new_subset=[]
                A[s_ind] = alpha * sum(np.diag(Tnew)) + (1 - alpha) * (sum(sum(Tnew)) - sum(np.diag(Tnew)));

        max_ind = np.argmax(A)
        maxval = A[max_ind]
        Subset_joint.append(index_set[max_ind])
    return Subset_joint



# snmi = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\SNMI.mat')
# cl = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\T.mat')
# pred = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\PRED.mat')
# #
# SNMI=snmi['SNMI']
# PRED=pred['PRED']
# T=cl['T']
# bestSize=3
# joint_criterion(bestSize,PRED,T,SNMI)

# PRED = genfromtxt('PRED.csv', int, delimiter=',')
# T = genfromtxt('T.csv', delimiter=',')
# SNMI = genfromtxt('SNMI.csv', delimiter=',')
# bestSize=3
# joint_criterion(bestSize,PRED,T,SNMI)

