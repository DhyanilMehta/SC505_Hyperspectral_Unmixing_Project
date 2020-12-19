import numpy as np
import time
from sklearn.decomposition import NMF

# NMF decomposition. A = WH
def et_NMF(A, r, er_out=False,W_prev=None, H_prev=None):	
	t_in = time.time()
	if W_prev is not None and H_prev is not None:
		NMF_model = NMF(n_components=r, solver='mu', init='custom', max_iter=10000)
		W = NMF_model.fit_transform(A, W=W_prev, H=H_prev)
		H = NMF_model.components_
	else:
		NMF_model = NMF(n_components=r, solver='mu', init='nndsvda', max_iter=10000)
		W = NMF_model.fit_transform(A)
		H = NMF_model.components_

	if er_out:
		err = NMF_model.reconstruction_err_
	else:
		err = 0
	
	t = time.time() - t_in
	return W, H, err, t