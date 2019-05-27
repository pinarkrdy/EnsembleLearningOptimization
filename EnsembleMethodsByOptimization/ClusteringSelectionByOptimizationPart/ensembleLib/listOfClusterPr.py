import numpy as np
import scipy.io
from sklearn.cluster import KMeans
from sklearn import metrics
import random

# This file generates the cluster pool with 3 different generation
#     1)Random initilization of clustering methods
#     2)Random feature selection
#     3)Random projection of Data Matrix
# List of clusters
# K-means clustering with different parameters"
# Parameters
#     ----------
#     c : int
#     ensembleSize : weights of cluster_runs in a list
#     X :array of shape - dataset
#     normalize_status : int
#     Returns
#     -------
#     PRED : prediction result that is returning from diferent kmeans intitlization
#     T: This is the matrix of Accuracy Diversity Matrix
#     SNMI : Sum of normalize mutual information
#     nmi mutual info score on prediction matrix
#
def listOfClusterPr(c,ensembleSize,X,normalize_status):


    rowLen = len(X)
    colLen = len(X[0])

    if normalize_status==1:
        X=np.divide(X,(np.outer(np.ones((rowLen,1)),np.sqrt(np.sum(np.square(X), axis=0)))))
    if normalize_status==2:
        aa=np.transpose(np.sqrt(np.sum(np.square(np.transpose(X)), axis=0)))
        bb=np.ones((1, colLen))
        cc=np.outer(aa, bb)
        X=np.divide(X,cc)

    #1)different random initializations%% ----NOTE: Different clustering
    pred1Temp=[]
    r=np.random.randint(1,(c+1), size=ensembleSize)
    for i in range(ensembleSize):
        pred1Temp.append(KMeans(n_clusters=r[i], random_state=0).fit(X).labels_)
    pred1=np.transpose(np.asarray(pred1Temp))

    #2) Choose random feature subsets
    pred2Temp=[]
    d=np.round(colLen/2)
    rf =np.random.randint(2,d+1,size=ensembleSize)
    for i in range(ensembleSize):
        resultX = X[:, np.random.permutation(np.int(rf[i])+1)]
        pred2Temp.append(KMeans(n_clusters=r[i], random_state=0).fit(resultX).labels_)
    pred2=np.transpose(np.asarray(pred2Temp))


    #3) Choose different random linear projection%%%
    pred3Temp=[]

    for m in range(ensembleSize):
        list = []
        list2 = []
        my_list = [(1 / np.sqrt(rf[m])), (-1 / np.sqrt(rf[m]))]
        my_samp = rf[m]
        for i in range(colLen):
            list = []
            for j in range(my_samp):
                list.append(random.choice(my_list))
            list2.append(list)


        R = np.asarray(list2)
        A = np.dot(X,R)
        pred3Temp.append(KMeans(n_clusters=r[m], random_state=0).fit(A).labels_)
    pred3=np.transpose(np.asarray(pred3Temp))
    PRED=np.concatenate((pred1,pred2,pred3), axis=1)


    # This is the matrix of Accuracy Diversity Matrix in ppt as G
    T=np.empty([(ensembleSize*3), (ensembleSize*3)]);
    nmi=np.empty([(ensembleSize*3), (ensembleSize*3)])
    SNMI=[0 for x in range((ensembleSize*3))]
    for i in range (ensembleSize*3):
        for j in range(ensembleSize*3):
            xxx=PRED[:, i]
            nmi[i,j] = metrics.mutual_info_score(PRED[:,i],PRED[:,j]);
            T[i,j]=1- nmi[i,j]

        SNMI[i]=np.sum(nmi[i,:])-nmi[i,i]
        T[i,i]=SNMI[i];

    aaa=1
    return PRED,T,SNMI,nmi