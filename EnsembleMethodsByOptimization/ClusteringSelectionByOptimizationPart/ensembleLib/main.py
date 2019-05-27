import numpy as np
import pandas as pd
import scipy.io
from numpy import genfromtxt
from EnsembleMethodsByOptimization.ClusteringSelectionByOptimizationPart.ensembleLib import listOfClusterPr, clusterEnsemble, evalMutual, ClusterEnsembleTest, \
    joint_criterion, JointTest, Cluster_select, ClusterSelectTest
from EnsembleMethodsByOptimization.ClusteringSelectionByOptimizationPart.ensembleLib.SpectralJordan import SpectralJordan

from EnsembleMethodsByOptimization.ClusteringSelectionByOptimizationPart.ensembleLib import Random_select


def main():
    ensembleSize=50
    normalize_status=2
    c=8
    X = scipy.io.loadmat('...\scikit-feature-master\skfeature\data\Yale.mat')
    X = X['yeast']  # data
    X = X.astype(float)
    rowLen = len(X)
    colLen = len(X[0])
    Y=X[:,colLen-1]
    Y=Y.astype(int)
    X= X[:, :-1]
    c = 8

    PRED, T, SNMI, nmi=listOfClusterPr.listOfClusterPr(c, ensembleSize, X, normalize_status)
    np.savetxt("PRED.csv", PRED, delimiter=",")
    np.savetxt("T.csv", T, delimiter=",")
    np.savetxt("SNMI.csv", SNMI, delimiter=",")
    np.savetxt("nmi.csv", nmi, delimiter=",")

    PRED = genfromtxt('PRED.csv',int, delimiter=',')
    T = genfromtxt('T.csv', delimiter=',')
    SNMI = genfromtxt('SNMI.csv', delimiter=',')
    nmi = genfromtxt('nmi.csv', delimiter=',')
    cls_full=PRED

    cl_full, qual_full= clusterEnsemble.clusterEnsemble(np.transpose(cls_full), c)
    np.savetxt("cl_full.csv", cl_full, delimiter=",")

    cl_full = genfromtxt('cl_full.csv',int, delimiter=',')

    MutualInfoFull=evalMutual.evalMutual(Y,cl_full)
    Try_num = 5
    bestMutualOPT=np.zeros((Try_num,2))
    MutualInfoJoint=np.zeros((Try_num,2))
    mutualInfoClusterSelect=np.zeros((Try_num,2))
    MutualInfoRand=np.zeros((Try_num,2))
    bestSize = np.zeros((Try_num, 2))
    BestSizeAll = np.zeros((Try_num, 2))

    for i in range(Try_num):
        for j in range(1,2):
            # #BIZIMKI
            bestMutualOPT[i,j], bestSize[i,j]=ClusterEnsembleTest.ClusterEnsembleTest(PRED,T,Y,c,j)
            #
            # #JOINT
            Subset_joint=joint_criterion.joint_criterion(bestSize[i,j],PRED,T,SNMI)
            np.savetxt("Subset_joint.csv", Subset_joint, delimiter=",")
            #
            Subset_joint = genfromtxt('Subset_joint.csv', int, delimiter=',')
            MutualInfoJoint[i,j]= JointTest.JointTest(Y, PRED, Subset_joint, c)

            #CLUSTER_SELECT

            # pred = scipy.io.loadmat('C:\MATLAB\mfilesICML\EnsembleClustering\ClusterEnsemble-V2.0\PRED.mat')
            # PRED = pred['PRED']
            # bestSize = 50
            # Nmi = scipy.io.loadmat('C:/MATLAB\mfilesICML/EnsembleClustering/ClusterEnsemble-V2.0/nmi.mat')
            # SNMI_ = scipy.io.loadmat('C:/MATLAB\mfilesICML/EnsembleClustering/ClusterEnsemble-V2.0/SNMI.mat')
            # nmi=Nmi['nmi']
            #
            # SNMI=SNMI_['SNMI'][0]
            IDX=SpectralJordan(bestSize[i,j],PRED)
            # #bestSize[i, j]

            Ensemble_subset=Cluster_select.Cluster_select(IDX,PRED,nmi,SNMI,bestSize[i,j])
            np.savetxt("Ensemble_subset.csv", cl_full, delimiter=",")
            Ensemble_subset = genfromtxt('Ensemble_subset.csv', int, delimiter=',')
            mutualInfoClusterSelect[i,j] = ClusterSelectTest.ClusterSelectTest(Y, PRED, Ensemble_subset, c)

            #RANDOM
            MutualInfoRand[i,j] = Random_select.Random_select(Y, c, bestSize[i, j], PRED)

            BestSizeAll[i,j]=bestSize[i,j]

    bestMutualOPTList = ",".join(bestMutualOPT.astype(str).tolist())
    MutualInfoJointList = ",".join(MutualInfoJoint.astype(str).tolist())
    mutualInfoClusterSelectList = ",".join(mutualInfoClusterSelect.astype(str).tolist())
    MutualInfoRandList = ",".join(MutualInfoRand.astype(str).tolist())
    BestSizeAllList = ",".join(BestSizeAll.astype(str).tolist())

    df = pd.DataFrame({"a": bestMutualOPT, "b": MutualInfoJoint, "c": mutualInfoClusterSelect,"d":MutualInfoRand,"e":BestSizeAll})
    writer = pd.ExcelWriter("result.xlsx")
    df.to_excel(writer, startrow=4, startcol=0)


main()