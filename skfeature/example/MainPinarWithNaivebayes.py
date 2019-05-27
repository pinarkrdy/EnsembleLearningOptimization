import numpy as np
import scipy.io
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB

from EnsembleMethodsByOptimization.dccp import dccpFunc
from skfeature.function.sql import SqlLiteDatabase

multiClass = True
x = np.random.randn(20)
# sq =SqlLiteDatabase.SqlLiteDatabase('C:\Users\lenovo\Desktop\scikit-feature-master\skfeature\FeatureSelectionDBColon_100.db')
# sq =SqlLiteDatabase.SqlLiteDatabase('C:\Users\lenovo\Desktop\scikit-feature-master\skfeature\FeatureSelectionD_warpAR10P.db')
#sq =SqlLiteDatabase.SqlLiteDatabase('C:\Users\lenovo\Desktop\scikit-feature-master\skfeature\FeatureSelectionDB_Madelon.db')
#sq =SqlLiteDatabase.SqlLiteDatabase('C:\Users\lenovo\Desktop\scikit-feature-master\skfeature\FeatureSelectionDB_Yale_100.db')
sq =SqlLiteDatabase.SqlLiteDatabase('C:\Users\lenovo\Desktop\scikit-feature-master\skfeature\FeatureSelectionDB_Small_Lung_100.db')




# BBB='CIFE_'+','.join(str(x) for x in x)
# #xstr = "".join(x_arrstr)
# sq.write('SelectedFeature', 'features',BBB)
# bbb=sq.get('SelectedFeature','features')
# load data
#mat = scipy.io.loadmat('../data/colon.mat')
#mat = scipy.io.loadmat('../data/madelon.mat')
#mat = scipy.io.loadmat('../data/warpAR10P.mat')
# mat = scipy.io.loadmat('../data/Yale.mat')
mat = scipy.io.loadmat('../data/lung_small.mat')
X = mat['X']  # data
X = X.astype(float)
# X=X[1:61]
numberofrow=X.shape[0]
y = mat['Y']  # label
# y=y[1:61]
y = y[:, 0]
n_samples, n_features = X.shape  # number of samples and number of features
matrix=np.zeros((n_features, 1))
num_fea=100


# for s in range(4,5):
#     numberofrow = X.shape[0]
#     a = range(0, numberofrow)
#     a = np.asarray(a).astype(float)
#     X = np.c_[a, X]
#
#     X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=s+1)
#     X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=s+1)
#     X_trainlabel=X_train[:,0].astype(int)
#     X_testlabel=X_test[:,0].astype(int)
#     X_vallabel=X_val[:,0].astype(int)
#     X_train=X_train[:,1:X_train.shape[1]]
#     X_val = X_val[:, 1:X_val.shape[1]]
#     X_test = X_test[:, 1:X_test.shape[1]]
#     line_ = 'Xtrain_'+str(s) +'_'+ ','.join(str(X_trainlabel) for X_trainlabel in X_trainlabel)
#     line = ''.join(c for c in line_ if c not in '[]. ')
#     sq.write('Xtrain', 'selected', line)
#     line = ''
#     line_ = ''
# #
#     line_ = 'Xtest_'+str(s)  +'_'+ ','.join(str(X_testlabel) for X_testlabel in X_testlabel)
#     line = ''.join(c for c in line_ if c not in '[]. ')
#     sq.write('Xtest', 'selected', line)
#     line = ''
#     line_ = ''
#
#     line_ = 'Xval_'+str(s) +'_'+ ','.join(str(X_vallabel) for X_vallabel in X_vallabel)
#     line = ''.join(c for c in line_ if c not in '[]. ')
#     sq.write('Xval', 'selected', line)
#     line = ''
#     line_ = ''

    # # #CIFE######################################################################################
    # idCFE, _, _ = CIFE.cife(X_train, y_train, n_selected_features=num_fea)
    # matrix[idCFE[0:num_fea],0 ]=1
    # featuresCFE =matrix
    # line_='CIFE_'+','.join(str(featuresCFE) for featuresCFE in featuresCFE)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    #
    # CMIM######################################################################################
    # idCMIM, _, _ = CMIM.cmim(X_train, y_train, n_selected_features=num_fea)
    # # obtain the dataset on the selected features
    # matrix[idCMIM[0:num_fea],0 ]=1
    # featuresCMIM = matrix
    # line_='CMIM_'+','.join(str(featuresCMIM) for featuresCMIM in featuresCMIM)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # #DISR######################################################################################
    # idDISR, _, _ = DISR.disr(X_train, y_train, n_selected_features=num_fea)
    # matrix[idDISR[0:num_fea],0 ]=1
    # featuresDISR = matrix
    # line_='DISR_'+','.join(str(featuresDISR) for featuresDISR in featuresDISR)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # #FCBF######################################################################################
    # idFCBF = FCBF.fcbf(X_train, y_train, n_selected_features=num_fea)
    # idFCBF= idFCBF[0]
    # matrix[idFCBF[0:num_fea],0 ]=1
    # featuresFCBF = matrix
    # line_='FCBF_'+','.join(str(featuresFCBF) for featuresFCBF in featuresFCBF)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # ICAP######################################################################################
    # idICAP, _, _ =ICAP.icap(X_train, y_train, n_selected_features=num_fea)
    # matrix[idICAP[0:num_fea],0 ]=1
    # featuresICAP = matrix
    # line_='ICAP_'+','.join(str(featuresICAP) for featuresICAP in featuresICAP)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    # # JMI######################################################################################
    # idJMI, _, _ =JMI.jmi(X_train, y_train, n_selected_features=num_fea)
    # matrix[idJMI[0:num_fea],0 ]=1
    # featuresJMI = matrix
    # line_='JMI_'+','.join(str(featuresJMI) for featuresJMI in featuresJMI)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # MIFS######################################################################################
    # idMIFS, _, _ =MIFS.mifs(X_train, y_train, n_selected_features=num_fea)
    # matrix[idMIFS[0:num_fea],0 ]=1
    # featuresMIFS = matrix
    # line_='MIFS_'+','.join(str(featuresMIFS) for featuresMIFS in featuresMIFS)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # MIM######################################################################################
    # idMIM, _, _ =MIM.mim(X_train, y_train, n_selected_features=num_fea)
    # matrix[idMIM[0:num_fea],0 ]=1
    # featuresMIM = matrix
    # line_='MIM_'+','.join(str(featuresMIM) for featuresMIM in featuresMIM)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # MRMR######################################################################################
    # idMRMR, _, _ =MRMR.mrmr (X_train, y_train, n_selected_features=num_fea)
    # matrix[idMRMR[0:num_fea],0 ]=1
    # featuresMRMR = matrix
    # line_='MRMR_'+','.join(str(featuresMRMR) for featuresMRMR in featuresMRMR)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # #SORUN YOK CFS######################################################################################
    # idCFS=CFS.cfs(X_train, y_train)
    # matrix[idCFS[0:num_fea],0 ]=1
    # featuresCFS = matrix
    # line_='CFS_'+','.join(str(featuresCFS) for featuresCFS in featuresCFS)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # #gini_index######################################################################################
    # scoregini_index =gini_index.gini_index (X_train, y_train)
    # idgini_index = gini_index.feature_ranking(scoregini_index)
    # matrix[idgini_index[0:num_fea],0 ]=1
    # featuresgini_index = matrix
    # line_='giniindex_'+','.join(str(featuresgini_index) for featuresgini_index in featuresgini_index)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # f_score######################################################################################
    # scoref_score=f_score.f_score(X_train, y_train)
    # idf_score = f_score.feature_ranking(scoref_score)
    # matrix[idf_score[0:num_fea],0 ]=1
    # featuresf_score =matrix
    # line_='fscore_'+','.join(str(featuresf_score) for featuresf_score in featuresf_score)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    # # fisher_score######################################################################################
    # scorefisher_score = fisher_score.fisher_score(X_train, y_train)
    # idfisher_score = fisher_score.feature_ranking(scorefisher_score)
    # matrix[idfisher_score[0:num_fea],0 ]=1
    # selected_featuresfisher_score =matrix
    # line_='fisherscore_'+','.join(str(selected_featuresfisher_score) for selected_featuresfisher_score in selected_featuresfisher_score)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # lap_score######################################################################################
    # # construct affinity matrix
    # kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    # W = construct_W.construct_W(X_train, **kwargs_W)
    # scorelap_score = lap_score.lap_score(X_train, W=W)
    # idlap_score = lap_score.feature_ranking(scorelap_score)
    # matrix[idlap_score[0:num_fea],0 ]=1
    # selected_featureslap_score = matrix
    # line_='lapscore_'+','.join(str(selected_featureslap_score) for selected_featureslap_score in selected_featureslap_score)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # ssss= np.column_stack((selected_featureslap_score,selected_featuresfisher_score))
    # # reliefF######################################################################################
    # scorereliefF = reliefF.reliefF(X_train, y_train)
    # idreliefF = reliefF.feature_ranking(scorereliefF)
    # matrix[idreliefF[0:num_fea],0 ]=1
    # selected_featuresreliefF =matrix
    # line_='relief_'+','.join(str(selected_featuresreliefF) for selected_featuresreliefF in selected_featuresreliefF)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # SPEC######################################################################################
    # # specify the second ranking function which uses all except the 1st eigenvalue
    # kwargs = {'style': 0}
    #
    # scoreSPEC = SPEC.spec(X_train, **kwargs)
    # idSPEC = SPEC.feature_ranking(scoreSPEC, **kwargs)
    # matrix[idSPEC[0:num_fea],0 ]=1
    # selected_featuresSPEC = matrix
    # line_='SPEC_'+','.join(str(selected_featuresSPEC) for selected_featuresSPEC in selected_featuresSPEC)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # # trace_ratio######################################################################################
    # idtrace_ratio, feature_score, subset_score = trace_ratio.trace_ratio(X_train, y_train, num_fea, style='fisher')
    # matrix[idtrace_ratio[0:num_fea],0 ]=1
    # selected_featurestrace_ratio =matrix
    # line_='traceratio_'+','.join(str(selected_featurestrace_ratio) for selected_featurestrace_ratio in selected_featurestrace_ratio)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # ll_l21######################################################################################
    # X = X_train   # data
    #
    # y = y_train    # label
    # Y = construct_label_matrix_pan(y)
    # Weight, obj, value_gamma = ll_l21.proximal_gradient_descent(X, Y, 0.1, verbose=False)
    # idll_l21 = feature_ranking(Weight)
    # matrix[idll_l21[0:num_fea],0 ]=1
    # selected_featuresll_l21 =matrix
    # line_='lll21_'+','.join(str(selected_featuresll_l21) for selected_featuresll_l21 in selected_featuresll_l21)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # MCFS######################################################################################
    # kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
    # W = construct_W.construct_W(X_train, **kwargs)
    # Weight = MCFS.mcfs(X_train, n_selected_features=num_fea, W=W, n_clusters=20)
    # idMCFS = MCFS.feature_ranking(Weight)
    # matrix[idMCFS[0:num_fea],0 ]=1
    # selected_featuresMCFS = matrix
    # line_='MCFS_'+','.join(str(selected_featuresMCFS) for selected_featuresMCFS in selected_featuresMCFS)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # NDFS######################################################################################
    # kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
    # W = construct_W.construct_W(X_train, **kwargs)
    # Weight = NDFS.ndfs(X_train, W=W, n_clusters=20)
    # idNDFS = feature_ranking(Weight)
    # matrix[idNDFS[0:num_fea],0 ]=1
    # selected_featuresNDFS = matrix
    # line_='NDFS_'+','.join(str(selected_featuresNDFS) for selected_featuresNDFS in selected_featuresNDFS)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # RFS######################################################################################
    # Y = construct_label_matrix(y_train)
    # Weight = RFS.rfs(X_train, Y, gamma=0.1)
    # idRFS = feature_ranking(Weight)
    # matrix[idRFS[0:num_fea],0 ]=1
    # selected_featuresRFS = matrix
    # line_='RFS_'+','.join(str(selected_featuresRFS) for selected_featuresRFS in selected_featuresRFS)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # UDFS######################################################################################
    # num_cluster = 20  # number of clusters, it is usually set as the number of classes in the ground truth
    # Weight = UDFS.udfs(X_train, gamma=0.1, n_clusters=num_cluster)
    # idUDFS = feature_ranking(Weight)
    # matrix[idUDFS[0:num_fea],0 ]=1
    # selected_featuresUDFS = matrix
    # line_='UDFS_'+','.join(str(selected_featuresUDFS) for selected_featuresUDFS in selected_featuresUDFS)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))

    # # alpha_investing######################################################################################
    # idalpha_investing = alpha_investing.alpha_investing(X_train, y_train, 0.05, 0.05)
    # matrix[idalpha_investing[0:num_fea],0 ]=1
    # selected_featuresalpha_investing = matrix
    # line_='alphainvesting_'+','.join(str(selected_featuresalpha_investing) for selected_featuresalpha_investing in selected_featuresalpha_investing)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
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
    #
    # # perform feature selection and obtain the feature weight of all the features
    # w, obj, value_gamma = group_fs.group_fs(X_train, y_train, z1, z2, idgroup_fs, verbose=True)
    # idx_ = np.argsort(w, 0)
    # idgroup_fs = idx_[::-1]
    #
    # matrix[idgroup_fs[0:num_fea],0 ]=1
    # selected_featuresgroup_fs = matrix
    # line_='groupfs_'+','.join(str(selected_featuresgroup_fs) for selected_featuresgroup_fs in selected_featuresgroup_fs)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
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
    # w, obj, value_gamma = tree_fs.tree_fs(X_train, y_train, z, idtree_fs, verbose=True)
    # idx2_ = np.argsort(w, 0)
    # idtree_fs = idx2_[::-1]
    # matrix[idtree_fs[0:num_fea],0 ]=1
    # selected_featurestree_fs = matrix
    # line_='treefs_'+','.join(str(selected_featurestree_fs) for selected_featurestree_fs in selected_featurestree_fs)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # decision_tree_backward ######################################################################################
    # iddecision_tree_backward = decision_tree_backward.decision_tree_backward(X_train, y_train, num_fea)
    # matrix[iddecision_tree_backward[0:num_fea],0 ]=1
    # X_selected = matrix
    # line_='decisiontreebackward_'+','.join(str(X_selected) for X_selected in X_selected)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # decision_tree_forward ######################################################################################
    # iddecision_tree_forward = decision_tree_forward.decision_tree_forward(X_train, y_train, num_fea)
    # matrix[iddecision_tree_forward[0:num_fea], 0] = 1
    # X_selecteddecision_tree_forward =matrix
    # line_='decisiontreeforward_'+','.join(str(X_selecteddecision_tree_forward) for X_selecteddecision_tree_forward in X_selecteddecision_tree_forward)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # svm_backward######################################################################################
    # idsvm_backward = svm_backward.svm_backward(X_train, y_train, num_fea)
    # matrix[idsvm_backward[0:num_fea],0 ]=1
    # X_selectedsvm_backward = matrix
    # line_='svmbackward_'+','.join(str(X_selectedsvm_backward) for X_selectedsvm_backward in X_selectedsvm_backward)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))
    #
    # # svm_forward######################################################################################
    # idsvm_forward = svm_forward.svm_forward(X_train, y_train, num_fea)
    # matrix[idsvm_forward[0:num_fea],0 ]=1
    # X_selectedsvm_forward = matrix
    # line_='svmforward_'+','.join(str(X_selectedsvm_forward) for X_selectedsvm_forward in X_selectedsvm_forward)
    # line=''.join( c for c in line_ if  c not in '[]. ' )
    # sq.write('FeatureSelectionResults'+str(s), 'features',line)
    # line=''
    # line_=''
    # matrix=np.zeros((n_features, 1))

#
# #
# #

FinalAccuracyValMean=[]
for s in range(5):
    error_rate_val = []
    accuracy_val = []
    error_rate_test = []
    accuracy_test = []
    # resultTest = []
    # resultVal = []
    # resultTrain = []
    train_list=''
    resultTrain=sq.get('XTrain' , 'selected')[s]
    resultTrain=''.join(resultTrain).encode('utf-8').decode('unicode-escape')

    train_list=np.asarray(resultTrain.split(",")).astype(int)
    trainResult=X[train_list,:]
    trainY=y[train_list]
    resultTest=sq.get('XTest' , 'selected')[s]
    resultTest = ''.join(resultTest).encode('utf-8').decode('unicode-escape')
    test_list=np.asarray(resultTest.split(",")).astype(int)
    testResult=X[test_list,:]
    testY=y[test_list]

    resultVal=sq.get('XVal', 'selected')[s]
    resultVal = ''.join(resultVal).encode('utf-8').decode('unicode-escape')
    val_list=np.asarray(resultVal.split(",")).astype(int)

    valResult=X[val_list,:]
    valY=y[val_list]
    results=sq.get('FeatureSelectionResults'+str(s), 'features')
    nresults=np.asarray(results);
    l2 = []

    for l in results:
        line_temp=str(l).split("_")[1]
        line_temp = str(line_temp).split("',)")[0]
        my_list =line_temp.split(",")
        l2.append(my_list)
        nparrayls = np.asarray(l2)
    resultMatrix=np.array(l2)

    # matrix de satirlar her bir feature selection algoritmasini, sutunlar her bir feature'i ifade ediyor
    rowLen = len(resultMatrix)
    colLen = len(resultMatrix[0])
    resultMatrix=resultMatrix.astype(int)
    T=np.zeros([rowLen, rowLen]);
    for i in range(rowLen):
        for j in range(rowLen):

            selectedindex = np.where(resultMatrix[i, :] == 1)[0]
            prunedX = trainResult[:, selectedindex]
            X_train1, X_test1, y_train1, y_test1 = train_test_split(prunedX, trainY, test_size=0.4)
            # svm classification
            gnb = GaussianNB()
            if multiClass == False:

                clf = gnb.fit(X_train1, y_train1)
            else:
                clf = gnb.fit(X_train1, y_train1)

            y_predicted = clf.predict(X_test1)
            if i == j:
                T[i,i]= accuracy_score(y_test1, y_predicted)
            else:
                selectedindex2 = np.where(resultMatrix[j, :] == 1)[0]
                prunedX_ = trainResult[:, selectedindex]
                X_train1, X_test1, y_train1, y_test1 = train_test_split(prunedX_, trainY, test_size=0.4)
                # svm classification
                if multiClass==False:
                    clf = gnb.fit(X_train1, y_train1)
                else:
                    clf = gnb.fit(X_train1, y_train1)

                y_predicted_ = clf.predict(X_test1)
                addedlistCount = np.sum( y_predicted_!= y_predicted)
                T[i, j]=addedlistCount

    #rho_list = [10, 100,10**-1,10**-2]#0.8 0.7 0.866666666667 0.6 0.466666666667
    #
                #rho_list = [10 **-1,10 **-2,10 **1,10**2]  # 0.8 0.7 0.866666666667 0.6 0.466666666667
     #rho_list = [10 **1,10 **2,10 **3,10 **-1,10 **-2,10 **-3] #warp
    rho_list = [10 ** 1, 10 ** 2, 10 ** 3]
    dccpResults= []
    sumofColumns=[]

    #DCCP ile hesap yap
    for index, rho in enumerate(rho_list):
        A= dccpFunc.dccpFunc(T, rho)
        if A!=[]:
            dccpResults.append(A)
    selectedFeatureSelectionAlg=[]
    len_res=len(dccpResults)

    FinalAccuracyVal = []
    FinalErrorVal = []
    for index in range(len_res):
        temp = dccpResults[index];
        temp= np.asarray(dccpResults[index]).astype(int)
        selectedFeatureSelectionAlg=resultMatrix[temp,:]
        length=int(round(len(temp)/3))
        sumofColumns=selectedFeatureSelectionAlg.sum(axis=0)
        selectedFeature = []
        for l, value in enumerate(sumofColumns):
            if value>length:
                selectedFeature.append(l)
        #aSGDAKI X VALIATION SET OLUCAK
        prunedX2 = []
        y_predicted2 = []
        prunedX2 =  valResult[:,selectedFeature]


        loo = LeaveOneOut()
        loo.get_n_splits(prunedX2)
        accuracy_val=[]
        error_rate_val=[]

        for train_index, test_index in loo.split(prunedX2):
            X_train2, X_test2 = prunedX2[train_index], prunedX2[test_index]
            y_train2, y_test2 = valY[train_index],valY[test_index]
            # svm classification
            if len(np.unique(y_train2))==1:
                if np.unique(y_train2)==1:
                    y_train2[0]=-1
                else:
                    y_train2[0] = 1
            gnb = GaussianNB()
            if multiClass == False:
                clf = gnb.fit(X_train2, y_train2)
            else:
                clf = gnb.fit(X_train2, y_train2)

            y_predicted2 = clf.predict(X_test2)
            accuracy=accuracy_score(y_test2, y_predicted2)
            accuracy_val.append(accuracy)
            error_rate_val.append(1 - accuracy)
        accuracy_val = np.asarray(accuracy_val)
        error_rate_val = np.asarray(error_rate_val)
        FinalAccuracyVal.append(sum(accuracy_val) / len(accuracy_val))
        FinalErrorVal.append(sum(error_rate_val) / len(error_rate_val))


    aaa=1
    selectedFeature = []
    sumofColumns=[]
    rho_index_min_error=(FinalErrorVal.index(min(FinalErrorVal)))
    temp = np.asarray(dccpResults[rho_index_min_error])
    selectedFeatureSelectionAlg=resultMatrix[temp,:]
    sumofColumns = selectedFeatureSelectionAlg.sum(axis=0)
    length = int(round(len(temp) /3))
    for index, value in enumerate(sumofColumns):
        if value > length:
            selectedFeature.append(index)
    prunedX3=[]
    y_predicted3=[]
    prunedX3 = testResult[:, selectedFeature]
    # X_train3, X_test3, y_train3, y_test3 = train_test_split(prunedX3, testY, test_size=0.2)
    # svm classification
    loo = LeaveOneOut()
    loo.get_n_splits(prunedX3)
    accuracy_test = []
    error_rate_test = []
    for train_index, test_index in loo.split(prunedX3):
        X_train3, X_test3 = prunedX3[train_index], prunedX3[test_index]
        y_train3, y_test3 = testY[train_index], testY[test_index]
        # svm classification
        gnb = GaussianNB()
        if multiClass == False:
            clf = gnb.fit(X_train3, y_train3)
        else:
            clf = gnb.fit(X_train3, y_train3)
        y_predicted3 = clf.predict(X_test3)
        accuracy = accuracy_score(y_test3, y_predicted3)
        accuracy_test.append(accuracy)
        error_rate_test.append(1 - accuracy)
    accuracy_test=np.asarray(accuracy_test)
    error_rate_test=np.asarray(error_rate_test)
    FinalAccuracy=sum(accuracy_test)/len(accuracy_test)
    FinalError= sum(error_rate_test)/len(error_rate_test)
    FinalAccuracyValMean.append(FinalAccuracy)
    print(FinalAccuracy)
FinalAccuracyValMean=np.asarray(FinalAccuracyValMean).astype(float)

print ("***********Final mean acc.****************")
print sum(FinalAccuracyValMean)/5
aa=1
#

# # #
# #
