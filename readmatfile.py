import json
import scipy.io
import numpy as np
mat = scipy.io.loadmat('dataset/Indian_pines.mat')
ctr = 0
for ele in mat['indian_pines']:
    # print(ele)
    ctr += 1
print(ctr, len(mat['indian_pines'][0]), len(mat['indian_pines'][0][0]))

print(np.asarray_chkfinite(mat['indian_pines']))