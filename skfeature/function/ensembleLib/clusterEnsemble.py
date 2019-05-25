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