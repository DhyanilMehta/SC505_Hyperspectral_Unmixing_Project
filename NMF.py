import numpy as np
import scipy.linalg as LA
import time
from sklearn.decomposition import non_negative_factorization, NMF

# NMF decomposition. A = WH
def et_NMF(A, r, k=0, er_out=False):	
	# random projection
	# t_in = time.time()
	# Y = np.abs(np.dot(np.random.randn(k, np.shape(A)[0]), A))
	# t_RP = time.time() - t_in

	# algorithm
	# W1, H1, _ = non_negative_factorization(Y, n_components=r)
	t_in = time.time()
	nmf = NMF(n_components=r, solver='mu', init='random', max_iter=1000)
	W = nmf.fit_transform(A)
	H = nmf.components_

	if er_out:
		err = nmf.reconstruction_err_
	else:
		err = 0
	
	t = time.time() - t_in
	return W, H, err, t