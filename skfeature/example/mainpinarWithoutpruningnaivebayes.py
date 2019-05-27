import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import MultinomialNB

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


#
gnb = MultinomialNB()
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


    sumofColumns=[]

    len_res=len(resultMatrix)

    FinalAccuracyVal = []
    FinalErrorVal = []

    length=int(round(len(resultMatrix)/3))
    sumofColumns=resultMatrix.sum(axis=0)
    selectedFeature = []
    for l, value in enumerate(sumofColumns):
        if value>length:
            selectedFeature.append(l)
        #aSGDAKI X VALIATION SET OLUCAK
    prunedX2 = []
    y_predicted2 = []
    prunedX2 =  testResult[:,selectedFeature]


    loo = LeaveOneOut()
    loo.get_n_splits(prunedX2)
    accuracy_val=[]
    error_rate_val=[]

    for train_index, test_index in loo.split(prunedX2):
        X_train2, X_test2 = prunedX2[train_index], prunedX2[test_index]
        y_train2, y_test2 = testY[train_index],testY[test_index]
            # svm classification
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
    print FinalAccuracyVal
    FinalAccuracyValMean.append(FinalAccuracyVal)
FinalAccuracyValMean=np.asarray(FinalAccuracyValMean).astype(float)

print ("***********Final mean acc.****************")
print sum(FinalAccuracyValMean)/5


# # #
# #
