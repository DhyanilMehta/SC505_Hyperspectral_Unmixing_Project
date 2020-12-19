import json
import scipy.io
import numpy as np
mat = scipy.io.loadmat('dataset/SalinasA_corrected.mat')
ctr = 0
print(mat.get('__globals__'))

print(np.asarray_chkfinite(mat['salinasA_corrected']).shape)

print(np.asarray_chkfinite(mat['salinasA_corrected']))