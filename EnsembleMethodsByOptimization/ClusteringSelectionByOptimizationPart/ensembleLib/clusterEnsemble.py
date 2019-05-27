# -*- coding: utf-8 -*-
import sys
from os import path
# sys.path.append(path.abspath('C:\Users\lenovo\Desktop\scikit-feature-master\myClustering-master'))
# import utils.load_dataset as ld
# import member_generation.library_generation as lg
# import utils.io_func as io
# import utils.settings as settings

import ensemble.Cluster_Ensembles as ce
import numpy as np
 # DESCRIPTION
 #   - performs the supra-consensus function for CLUSTER ENSEMBLES
 #     to combine multiple clustering label vectors e.g., cls = [cl1; cl2;...]
 #   - returns a row vector of the combined clustering labels
 #   - the following consensus function is  computed
 #     - HyperGraph Partitioning Algorithm (HGPA)
 #     and the one with the maximum average normalized mutual information
 #     is returned
 # ARGUMENTS
 #   cls   - matrix of labelings, one labeling per row (n x p)
 #           entries must be integers from 1 to k1 (row 1), 1 to k2 (row 2),...
 #           use NaN as an entry for unknown/missing labels
 #   k     - 1,2,3,... maximum number of clusters in the combined clustering
 #           (optional, default max(max(cls))
 #
 # REFERENCE
 #   please refer to the following paper if you use CLUSTER ENSEMBLES
 #     A. Strehl and J. Ghosh. "Cluster Ensembles - A Knowledge Reuse
	#   Framework for Combining Multiple Partitions", Journal on
	#   Machine Learning Research (JMLR), 3:583-617, December 2002

def clusterEnsemble(cls,k):
    # name = 'Iris'
    # d, t = ld.load_iris()
    #lib_name = lg.generate_library(d, t, name, 10, 3)
    # lib_name = lg.generate_library(d, t, name, 10, 3)

    lib = cls
    cluster_runs = np.random.randint(0, 50, (50, 5000))
    #ensemble_result = ce.cluster_ensembles(cluster_runs, verbose = True, N_clusters_max = 5)
    #ensemble_result = ce.cluster_ensembles_HGPAONLY(lib, N_clusters_max=2)
    ensemble_result = ce.cluster_ensembles_HGPAONLY(lib, N_clusters_max=k)
    #ensemble_result = ce.HGPA('./Cluster_Ensembles.h5', cluster_runs, verbose=True, N_clusters_max=50)
    #ensemble_result = ce.cluster_ensembles(cluster_runs,verbose = True, N_clusters_max = 3)
    # print metrics.normalized_max_mutual_info_score(t, ensemble_result)
    qual=ce.ceEvalMutual(cls,ensemble_result)
    return ensemble_result,qual