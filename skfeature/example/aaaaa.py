
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#madelon = 'http://archive.ics.uci.edu/ml/datasets/madelon'
mat = scipy.io.loadmat('../data/lung_small.mat')
X = mat['X']  # data
X = X.astype(float)
# X=X[1:61]
numberofrow=X.shape[0]
y = mat['Y']  # label

knn =KNeighborsClassifier(n_neighbors=4)



sbs = SFS(knn,
          k_features=4,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=0,
          verbose=2,
          #n_jobs=-1
          )
sbs=sbs.fit(X,y)

print(sbs.subsets_)
print(sbs.k_feature_idx_)
print('CV score: ')
print(sbs.k_score_)
aaa=1