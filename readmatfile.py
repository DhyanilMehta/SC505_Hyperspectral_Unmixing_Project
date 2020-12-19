import json
import scipy.io
import numpy as np
mat = scipy.io.loadmat('dataset/PaviaU.mat')
ctr = 0
print(mat.keys())

print(np.asarray_chkfinite(mat['paviaU']).shape)

print(np.asarray_chkfinite(mat['paviaU']))