import numpy as np
import scipy.linalg as LA
import time
from sklearn.decomposition import non_negative_factorization, NMF

# NMF decomposition. A = WH
def et_NMF(A, r, k=0, er_out=False,W_prev=None, H_prev=None):	
	# random projection
	# t_in = time.time()
	# Y = np.abs(np.dot(np.random.randn(k, np.shape(A)[0]), A))
	# t_RP = time.time() - t_in

	# algorithm
	# W1, H1, _ = non_negative_factorization(Y, n_components=r)
	t_in = time.time()
	if W_prev is not None and H_prev is not None:
		NMF_model = NMF(n_components=r, solver='mu', init='custom')
		W = NMF_model.fit_transform(A, W=W_prev, H=H_prev)
		H = NMF_model.components_
	else:
		NMF_model = NMF(n_components=r, solver='mu')
		W = NMF_model.fit_transform(A)
		H = NMF_model.components_

	if er_out:
		err = NMF_model.reconstruction_err_
	else:
		err = 0
	
	t = time.time() - t_in
	return W, H, err, t