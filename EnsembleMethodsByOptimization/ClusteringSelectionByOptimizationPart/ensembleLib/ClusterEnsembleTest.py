
import numpy as np
import scipy.optimize
from EnsembleMethodsByOptimization.dccp import dccpFunc
from EnsembleMethodsByOptimization.ClusteringSelectionByOptimizationPart.ensembleLib import evalMutual, objCluster
from EnsembleMethodsByOptimization.ClusteringSelectionByOptimizationPart.ensembleLib import clusterEnsemble
from sklearn.metrics import normalized_mutual_info_score
import scipy.optimize as op

# This file apply Disciplined Convex-Concave Programming(DCCP) on accurcay and diverstiy matrix(T) with specific Rho Parameters

# This test basically does the following
#   -Check the isDCCP value if it equals to 1 apply dccp algoritihm on T matrix ,otherwise the program will apply scipy.optimize
#     if it is zero then appy scipy.optimize library for optimize Prediction value
#   -Perform cluster ensemble algorithm
#   -Compute Mutual İnformation

# Parameters
#     ----------
#     T : int
#     PRED : weights of cluster_runs in a list
#     X :array of shape - dataset
#     Y : int
#     c : int
#     isDCCP : choosen method(1=dcccp; 0=scipy.optimize)
#     Returns
#     -------
#     PRED : prediction result that is returning from diferent kmeans intitlization
#     T: This is the matrix of Accuracy Diversity Matrix
#     SNMI : Sum of normalize mutual information
#     nmi : mutual info score on prediction matrix
def ClusterEnsembleTest(PRED,T,Y,c,isDCCP):
    #applymosek
    sizeEns=[]
    Quality=[]
    MutualInfo=[]
    if isDCCP==1:
        Rho = [10 ** -1, 10 ** -2, 10 ** -3, 1, 10]
        # Finds different pruning results for different rho values

        SList= dccpFunc.dccpFunc(T, Rho)
        #SList=[[5, 6, 8, 10, 12, 13, 16, 17, 19, 27, 31, 38, 45, 55, 59, 61, 62, 68, 72, 74, 75, 77, 79, 81, 82, 83, 87, 93, 94, 97, 105, 112, 113, 115, 119, 120, 124, 126, 128, 129, 130, 132, 134, 140, 141, 142], [], [2, 5, 6, 12, 19, 25, 28, 31, 38, 49, 51, 54, 55, 57, 58, 60, 63, 66, 67, 71, 74, 75, 76, 78, 85, 87, 90, 91, 92, 97, 98, 107, 110, 113, 117, 120, 122, 128, 130, 133, 139, 141, 142]]

        for S in SList:

            if len(S)!=0:
                sizeEns.append(len(S))
                Pruned=PRED[:,S]
                cn = Y
                cls = Pruned
                cl,qual= clusterEnsemble.clusterEnsemble(np.transpose(cls), c)
                Quality.append(qual)
                MutualInfo.append(evalMutual.evalMutual(cn, cl))
                # aaa=evalMutual.evalMutual(cn, cl)
                # bbb=normalized_mutual_info_score(cn, cl.reshape((1484,1)))

    else :
        # Rho = [10 ** -2, 10 ** -1,1, 10 ,100, 1000, 10000,100000]
        Rho = [10 ** -1, 10 ** -2, 10 ** -3]
        for rho in Rho:
            X0 = np.random.rand(150, 1)
            S = op.minimize(fun=objCluster.obj_cluster, x0=X0, args = (T,rho), method = 'TNC')
            optimal_res = S.x;
            S=optimal_res

            # S= scipy.minimize(func, [-1.0, 1.0], args=(-1.0,), jac=func_deriv, constraints=cons, method='Newton-CG', options={'disp': True})
            sizeEns.append(len(S))
            if len(S)!=0:
                Pruned=PRED[:,S]
                cn = Y
                cls = Pruned
                cl,qual= clusterEnsemble(np.transpose(cls), c)
                Quality.append(qual)
                MutualInfo.append(evalMutual(cn, cl))
    ind = np.argmax(Quality)
    maxval=Quality[ind]
    bestMutual = MutualInfo[ind]
    bestSize = sizeEns[ind]
    return bestMutual,bestSize



# trueclass = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\Y.mat')
# cl = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\T.mat')
# pred = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\PRED.mat')
# PRED=pred['PRED']
# T=cl['T']
# Y=trueclass['Y']
# isDCCP=1
# c=8
# ClusterEnsembleTest(PRED,T,Y,c,isDCCP)









