import numpy as np
x = np.array([[1,2,3], [0, 1, 0], [7, 0, 2]])
cc=x[:,0]
aaa=np.where(x[1,:] == 0)[0]
bbb=0