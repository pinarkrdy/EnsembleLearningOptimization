import scipy.io
from skfeature.function.information_theoretical_based import CIFE,CMIM,DISR,FCBF,ICAP,JMI,LCSI,MIFS,MIM,MRMR
from skfeature.function.statistical_based import CFS,chi_square,gini_index,f_score,low_variance,t_score
from skfeature.function.similarity_based import fisher_score,lap_score,reliefF,SPEC,trace_ratio
from skfeature.function.ensembleLib import clusterEnsemble,listOfClusterPr,evalMutual,deneme
from skfeature.function.sparse_learning_based import ll_l21,ls_l21,MCFS,NDFS,RFS,UDFS
from skfeature.function.streaming import alpha_investing
from skfeature.function.structure import graph_fs,group_fs,tree_fs
from skfeature.function.dccp import dccpFunc
from skfeature.function.wrapper import decision_tree_backward,decision_tree_forward,svm_backward,svm_forward
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import *
from skfeature.function.sql import SqlLiteDatabase
import json
import sqlite3 as sql
import numpy as np
from sklearn import metrics
import random

x = np.random.randn(20)
sq =SqlLiteDatabase.SqlLiteDatabase('C:\Users\lenovo\Desktop\FeatureSelectionDB.db')
# BBB='CIFE_'+','.join(str(x) for x in x)
# #xstr = "".join(x_arrstr)
# sq.write('SelectedFeature', 'features',BBB)
# bbb=sq.get('SelectedFeature','features')
# load data
mat = scipy.io.loadmat('../data/lung_small.mat')
X = mat['X']  # data
X = X.astype(float)
y = mat['Y']  # label
y = y[:, 0]
n_samples, n_features = X.shape  # number of samples and number of features
matrix=np.zeros((n_features, 1))
num_fea=20

# #CIFE######################################################################################
# idCFE, _, _ = CIFE.cife(X, y, n_selected_features=num_fea)
# matrix[idCFE[0:num_fea],0 ]=1
# featuresCFE =matrix
# line_='CIFE_'+','.join(str(featuresCFE) for featuresCFE in featuresCFE)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
#
# CMIM######################################################################################
# idCMIM, _, _ = CMIM.cmim(X, y, n_selected_features=num_fea)
# # obtain the dataset on the selected features
# matrix[idCMIM[0:num_fea],0 ]=1
# featuresCMIM = matrix
# line_='CMIM_'+','.join(str(featuresCMIM) for featuresCMIM in featuresCMIM)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# #DISR######################################################################################
# idDISR, _, _ = DISR.disr(X, y, n_selected_features=num_fea)
# matrix[idDISR[0:num_fea],0 ]=1
# featuresDISR = matrix
# line_='DISR_'+','.join(str(featuresDISR) for featuresDISR in featuresDISR)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# #FCBF######################################################################################
# idFCBF = FCBF.fcbf(X, y, n_selected_features=num_fea)
# idFCBF= idFCBF[0]
# matrix[idFCBF[0:num_fea],0 ]=1
# featuresFCBF = matrix
# line_='FCBF_'+','.join(str(featuresFCBF) for featuresFCBF in featuresFCBF)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # ICAP######################################################################################
# idICAP, _, _ =ICAP.icap(X, y, n_selected_features=num_fea)
# matrix[idICAP[0:num_fea],0 ]=1
# featuresICAP = matrix
# line_='ICAP_'+','.join(str(featuresICAP) for featuresICAP in featuresICAP)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
# # JMI######################################################################################
# idJMI, _, _ =JMI.jmi(X, y, n_selected_features=num_fea)
# matrix[idJMI[0:num_fea],0 ]=1
# featuresJMI = matrix
# line_='JMI_'+','.join(str(featuresJMI) for featuresJMI in featuresJMI)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # LCSI######################################################################################
# # idLCSI, _, _ =LCSI.lcsi(X, y, n_selected_features=num_fea)
# # matrix[idLCSI[0:num_fea],0 ]=1
# # featuresLCSI = matrix
# # MIFS######################################################################################
# idMIFS, _, _ =MIFS.mifs(X, y, n_selected_features=num_fea)
# matrix[idMIFS[0:num_fea],0 ]=1
# featuresMIFS = matrix
# line_='MIFS_'+','.join(str(featuresMIFS) for featuresMIFS in featuresMIFS)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # MIM######################################################################################
# idMIM, _, _ =MIM.mim(X, y, n_selected_features=num_fea)
# matrix[idMIM[0:num_fea],0 ]=1
# featuresMIM = matrix
# line_='MIM_'+','.join(str(featuresMIM) for featuresMIM in featuresMIM)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # MRMR######################################################################################
# idMRMR, _, _ =MRMR.mrmr (X, y, n_selected_features=num_fea)
# matrix[idMRMR[0:num_fea],0 ]=1
# featuresMRMR = matrix
# line_='MRMR_'+','.join(str(featuresMRMR) for featuresMRMR in featuresMRMR)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# #SORUN YOK CFS######################################################################################
# idCFS=CFS.cfs(X, y)
# matrix[idCFS[0:num_fea],0 ]=1
# featuresCFS = matrix
# line_='CFS_'+','.join(str(featuresCFS) for featuresCFS in featuresCFS)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # chi_square#####olmadi#################################################################################
# scorechi_square = chi_square.chi_square(X, y)
# # rank features in descending order according to score
# idchi_square = chi_square.feature_ranking(scorechi_square)
# matrix[idchi_square[0:num_fea],0 ]=1
# featureschi_square = matrix
# line_='chi_square_'+','.join(str(featureschi_square) for featureschi_square in featureschi_square)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # gini_index######################################################################################
# scoregini_index =gini_index.gini_index (X, y)
# idgini_index = gini_index.feature_ranking(scoregini_index)
# matrix[idgini_index[0:num_fea],0 ]=1
# featuresgini_index = matrix
# line_='gini_index_'+','.join(str(featuresgini_index) for featuresgini_index in featuresgini_index)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # f_score######################################################################################
# scoref_score=f_score.f_score(X, y)
# idf_score = f_score.feature_ranking(scoref_score)
# matrix[idf_score[0:num_fea],0 ]=1
# featuresf_score =matrix
# line_='f_score_'+','.join(str(featuresf_score) for featuresf_score in featuresf_score)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
# # t_score######################################################################################
# X = mat['X']  # data
# X = X.astype(float)
# y = mat['Y']  # label
# y = y[:, 0]
# score = t_score.t_score(X, y)
# idt_score = t_score.feature_ranking(score)
# matrix[idt_score[0:num_fea], 0] = 1
# selected_featurest_score =matrix
#
# line_='t_score_'+','.join(str(selected_featurest_score) for selected_featurest_score in selected_featurest_score)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # fisher_score######################################################################################
# scorefisher_score = fisher_score.fisher_score(X, y)
# idfisher_score = fisher_score.feature_ranking(scorefisher_score)
# matrix[idfisher_score[0:num_fea],0 ]=1
# selected_featuresfisher_score =matrix
# line_='fisher_score_'+','.join(str(selected_featuresfisher_score) for selected_featuresfisher_score in selected_featuresfisher_score)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # lap_score######################################################################################
# # construct affinity matrix
# kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
# W = construct_W.construct_W(X, **kwargs_W)
# scorelap_score = lap_score.lap_score(X, W=W)
# idlap_score = lap_score.feature_ranking(scorelap_score)
# matrix[idlap_score[0:num_fea],0 ]=1
# selected_featureslap_score = matrix
# line_='lap_score_'+','.join(str(selected_featureslap_score) for selected_featureslap_score in selected_featureslap_score)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# ssss= np.column_stack((selected_featureslap_score,selected_featuresfisher_score))
# # reliefF######################################################################################
# scorereliefF = reliefF.reliefF(X, y)
# idreliefF = reliefF.feature_ranking(scorereliefF)
# matrix[idreliefF[0:num_fea],0 ]=1
# selected_featuresreliefF =matrix
# line_='relief_'+','.join(str(selected_featuresreliefF) for selected_featuresreliefF in selected_featuresreliefF)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # SPEC######################################################################################
# # specify the second ranking function which uses all except the 1st eigenvalue
# kwargs = {'style': 0}
#
# scoreSPEC = SPEC.spec(X, **kwargs)
# idSPEC = SPEC.feature_ranking(scoreSPEC, **kwargs)
# matrix[idSPEC[0:num_fea],0 ]=1
# selected_featuresSPEC = matrix
# line_='SPEC_'+','.join(str(selected_featuresSPEC) for selected_featuresSPEC in selected_featuresSPEC)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# # trace_ratio######################################################################################
# idtrace_ratio, feature_score, subset_score = trace_ratio.trace_ratio(X, y, num_fea, style='fisher')
# matrix[idtrace_ratio[0:num_fea],0 ]=1
# selected_featurestrace_ratio =matrix
# line_='trace_ratio_'+','.join(str(selected_featurestrace_ratio) for selected_featurestrace_ratio in selected_featurestrace_ratio)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # ll_l21######################################################################################
# X = mat['X']    # data
# X = X.astype(float)
# y = mat['Y']    # label
# y = y[:, 0]
# Y = construct_label_matrix_pan(y)
# Weight, obj, value_gamma = ll_l21.proximal_gradient_descent(X, Y, 0.1, verbose=False)
# idll_l21 = feature_ranking(Weight)
# matrix[idll_l21[0:num_fea],0 ]=1
# selected_featuresll_l21 =matrix
# line_='ll_l21_'+','.join(str(selected_featuresll_l21) for selected_featuresll_l21 in selected_featuresll_l21)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

## ls_l21######################################################################################
# Weight, obj, value_gamma = ls_l21.proximal_gradient_descent(X, y, 0.1, verbose=False)
# idls_l21 = feature_ranking(Weight)
# matrix[idls_l21[0:num_fea],0 ]=1
# selected_featuresls_l21 = matrix
# line_='ls_l21_'+','.join(str(selected_featuresls_l21) for selected_featuresls_l21 in selected_featuresls_l21)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # MCFS######################################################################################
# kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
# W = construct_W.construct_W(X, **kwargs)
# Weight = MCFS.mcfs(X, n_selected_features=num_fea, W=W, n_clusters=20)
# idMCFS = MCFS.feature_ranking(Weight)
# matrix[idMCFS[0:num_fea],0 ]=1
# selected_featuresMCFS = matrix
# line_='MCFS_'+','.join(str(selected_featuresMCFS) for selected_featuresMCFS in selected_featuresMCFS)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # NDFS######################################################################################
# kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
# W = construct_W.construct_W(X, **kwargs)
# Weight = NDFS.ndfs(X, W=W, n_clusters=20)
# idNDFS = feature_ranking(Weight)
# matrix[idNDFS[0:num_fea],0 ]=1
# selected_featuresNDFS = matrix
# line_='NDFS_'+','.join(str(selected_featuresNDFS) for selected_featuresNDFS in selected_featuresNDFS)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # RFS######################################################################################
# Y = construct_label_matrix(y)
# Weight = RFS.rfs(X, Y, gamma=0.1)
# idRFS = feature_ranking(Weight)
# matrix[idRFS[0:num_fea],0 ]=1
# selected_featuresRFS = matrix
# line_='RFS_'+','.join(str(selected_featuresRFS) for selected_featuresRFS in selected_featuresRFS)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # UDFS######################################################################################
# num_cluster = 20  # number of clusters, it is usually set as the number of classes in the ground truth
# Weight = UDFS.udfs(X, gamma=0.1, n_clusters=num_cluster)
# idUDFS = feature_ranking(Weight)
# matrix[idUDFS[0:num_fea],0 ]=1
# selected_featuresUDFS = matrix
# line_='UDFS_'+','.join(str(selected_featuresUDFS) for selected_featuresUDFS in selected_featuresUDFS)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
#
# # alpha_investing######################################################################################
# idalpha_investing = alpha_investing.alpha_investing(X, y, 0.05, 0.05)
# matrix[idalpha_investing[0:num_fea],0 ]=1
# selected_featuresalpha_investing = matrix
# line_='alpha_investing_'+','.join(str(selected_featuresalpha_investing) for selected_featuresalpha_investing in selected_featuresalpha_investing)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # group_fs######################################################################################
# z1 = 0.1  # specify the regularization parameter of L1 norm
# z2 = 0.1  # specify the regularization parameter of L2 norm for the non-overlapping group
#
# # specify the group structure among features
# idgroup_fs = np.array([[1, 20, np.sqrt(20)], [21, 40, np.sqrt(20)], [41, 50, np.sqrt(10)],
#                 [51, 70, np.sqrt(20)], [71, 100, np.sqrt(30)]]).T
# idgroup_fs = idgroup_fs.astype(int)

# # perform feature selection and obtain the feature weight of all the features
# w, obj, value_gamma = group_fs.group_fs(X, y, z1, z2, idgroup_fs, verbose=True)
# idx_ = np.argsort(w, 0)
# idgroup_fs = idx_[::-1]
#
# matrix[idgroup_fs[0:num_fea],0 ]=1
# selected_featuresgroup_fs = matrix
# line_='group_fs_'+','.join(str(selected_featuresgroup_fs) for selected_featuresgroup_fs in selected_featuresgroup_fs)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))
# # tree_fs######################################################################################
# z = 0.01  # specify the regularization parameter of regularization parameter of L2 norm for the non-overlapping group
#
# # specify the tree structure among features
# idtree_fs = np.array([[-1, -1, 1], [1, 20, np.sqrt(20)], [21, 40, np.sqrt(20)], [41, 50, np.sqrt(10)],
#                 [51, 70, np.sqrt(20)], [71, 100, np.sqrt(30)], [1, 50, np.sqrt(50)], [51, 100, np.sqrt(50)]]).T
# idtree_fs = idtree_fs.astype(int)
#
# # perform feature selection and obtain the feature weight of all the features
# w, obj, value_gamma = tree_fs.tree_fs(X, y, z, idtree_fs, verbose=True)
# idx2_ = np.argsort(w, 0)
# idtree_fs = idx2_[::-1]
# matrix[idtree_fs[0:num_fea],0 ]=1
# selected_featurestree_fs = matrix
# line_='tree_fs_'+','.join(str(selected_featurestree_fs) for selected_featurestree_fs in selected_featurestree_fs)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # decision_tree_backward ######################################################################################
# iddecision_tree_backward = decision_tree_backward.decision_tree_backward(X, y, num_fea)
# matrix[iddecision_tree_backward[0:num_fea],0 ]=1
# X_selected = matrix
# line_='decision_tree_backward_'+','.join(str(X_selected) for X_selected in X_selected)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # decision_tree_forward ######################################################################################
# iddecision_tree_forward = decision_tree_forward.decision_tree_forward(X, y, num_fea)
# matrix[iddecision_tree_forward[0:num_fea], 0] = 1
# X_selecteddecision_tree_forward =matrix
# line_='decision_tree_forward_'+','.join(str(X_selecteddecision_tree_forward) for X_selecteddecision_tree_forward in X_selecteddecision_tree_forward)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # svm_backward######################################################################################
# idsvm_backward = svm_backward.svm_backward(X, y, num_fea)
# matrix[idsvm_backward[0:num_fea],0 ]=1
# X_selectedsvm_backward = matrix
# line_='svm_backward_'+','.join(str(X_selectedsvm_backward) for X_selectedsvm_backward in X_selectedsvm_backward)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

# # svm_forward######################################################################################
# idsvm_forward = svm_forward.svm_forward(X, y, num_fea)
# matrix[idsvm_forward[0:num_fea],0 ]=1
# X_selectedsvm_forward = matrix
# line_='svm_forward_'+','.join(str(X_selectedsvm_forward) for X_selectedsvm_forward in X_selectedsvm_forward)
# line=''.join( c for c in line_ if  c not in '[]. ' )
# sq.write('SelectedFeature', 'features',line)
# line=''
# line_=''
# matrix=np.zeros((n_features, 1))

results=sq.get('SelectedFeature', 'features')
l2 = []

for l in results:
    line_temp=str(l).split("_")[1]
    line_temp = str(line_temp).split("',)")[0]
    my_list =line_temp.split(",")
    l2.append(my_list)
resultMatrix=np.transpose(np.array(l2))

# matrix de satirlar her bir feature selection algoritmasini, sutunlar her bir feature'i ifade ediyor
rowLen = len(resultMatrix)
colLen = len(resultMatrix[0])

#Create T matrix with SNMI and NMI
T=np.empty([colLen, colLen]);
nmi=np.empty([colLen, colLen])
SNMI=[0 for x in range(colLen)]
for i in range (colLen):
    for j in range(colLen):
        nmi[i,j] = metrics.mutual_info_score(resultMatrix[:,i],resultMatrix[:,j]);
        T[i,j]=1- nmi[i,j]

    SNMI[i]=np.sum(nmi[i,:])-nmi[i,i]
    T[i,i]=SNMI[i];
########################################
cls_full=resultMatrix
c= np.unique(y).size
#CLS full gonderince ici 1 ve 0 lar oldugu icin calismior
cl_full =clusterEnsemble.clusterEnsemble(T,c)
A=dccpFunc.dccpFunc(T)
pruned=resultMatrix[:, A]
aaaaaAA=1
#MutualInfoFull = evalMutual.evalMutual(y,np.transpose(cl_full))
aaa=1
