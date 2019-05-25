# -*- coding: utf-8 -*-
# 引入相关模块
import utils.load_dataset as ld
import member_generation.library_generation as lg
import utils.io_func as io
import utils.settings as settings
import ensemble.Cluster_Ensembles as ce
import evaluation.Metrics as metrics
import numpy as np

name = 'Iris'
d, t = ld.load_iris()
#lib_name = lg.generate_library(d, t, name, 10, 3)
lib_name = lg.generate_library(d, t, name, 10, 3)

lib = io.read_matrix(settings.default_library_path + name + '/' + lib_name)

cluster_runs = np.random.randint(0, 50, (50, 5000))
#ensemble_result = ce.cluster_ensembles(cluster_runs, verbose = True, N_clusters_max = 5)
#ensemble_result = ce.cluster_ensembles_HGPAONLY(lib, N_clusters_max=2)
ensemble_result = ce.cluster_ensembles_HGPAONLY(lib, N_clusters_max=11)
#ensemble_result = ce.HGPA('./Cluster_Ensembles.h5', cluster_runs, verbose=True, N_clusters_max=50)
#ensemble_result = ce.cluster_ensembles(cluster_runs,verbose = True, N_clusters_max = 3)
print ensemble_result
print metrics.normalized_max_mutual_info_score(t, ensemble_result)
