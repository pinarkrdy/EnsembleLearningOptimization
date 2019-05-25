import scipy.io
import numpy as np
from sklearn.cluster import SpectralClustering
import scipy
import os
def SpectralJordan(k,data):
    data=np.transpose(data)
    clustering = SpectralClustering(n_clusters=k,
                                    random_state=None).fit(data)

    cccc=np.sort(clustering.labels_)
    return clustering.labels_
# pred = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\PRED.mat')
# PRED=pred['PRED']
# PRED=np.transpose(PRED)
# bestSize=3
# SpectralJordan(bestSize,PRED)