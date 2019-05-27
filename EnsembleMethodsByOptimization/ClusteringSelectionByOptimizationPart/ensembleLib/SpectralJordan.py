import scipy.io
import numpy as np
from sklearn.cluster import SpectralClustering
import scipy
import os
 # CONCEPT: Introduced the normalization process of affinity matrix(D-1/2 A D-1/2),
 # eigenvectors orthonormal conversion and clustering by kmeans
# Parameters
#     ----------
#     K : Number of cluster
#     data=array of shape - rediction result
#     Returns
#     -------
#     MutualInfo : mutual info score on prediction matrix


def SpectralJordan(k,data):
    data=np.transpose(data)
    clustering = SpectralClustering(n_clusters=k,
                                    random_state=None).fit(data)
    return clustering.labels_
# pred = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\PRED.mat')
# PRED=pred['PRED']
# PRED=np.transpose(PRED)
# bestSize=3
# SpectralJordan(bestSize,PRED)