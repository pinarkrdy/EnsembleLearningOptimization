import numpy as np

obj = np.random.randint(10, size=(3, 4, 5), dtype=np.int8)  # example array

with open('test.txt', 'wb') as f:
    np.savetxt(f, np.column_stack(obj), fmt='%1.10f')