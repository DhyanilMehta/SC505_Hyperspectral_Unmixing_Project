import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF

# NMF decomposition. A = WH
def calc_NMF(A, r, W_prev=None, H_prev=None):
	t_in = time.time()
	if W_prev is not None and H_prev is not None:
		NMF_model = NMF(n_components=r, solver='mu', init='custom', max_iter=1000)
		W = NMF_model.fit_transform(A, W=W_prev, H=H_prev)
		H = NMF_model.components_
	else:
		NMF_model = NMF(n_components=r, solver='mu', init='random', random_state=0, max_iter=1000)
		W = NMF_model.fit_transform(A)
		H = NMF_model.components_

	err = NMF_model.reconstruction_err_
	t = time.time() - t_in
	return W, H, err, t


# Plot calculated unmixed spectral endmembers (H) and error per iteration
def plot_NMF_data(name, endmember_names):
    data = np.load(f"data/nmf_{name}_data.npz")
    # W_l = data["W_list"]
    H_l = data["H_list"]

    H = H_l[len(H_l) - 1]
    wavelengths = np.linspace(0.4, 2.5, num=H.shape[1])

    plt.figure()
    for i in range(H_l[0].shape[0]):
        plt.plot(wavelengths, (H_l[0][i] * 500))
    plt.xlabel("Wavelength (in µm)")
    plt.ylabel("Relative Spectral Response")
    plt.legend(endmember_names, loc="best")


    err_data = data["error"]
    iters = np.arange(0, len(err_data), 1).tolist()
    plt.figure()
    plt.plot(iters, err_data.tolist())
    plt.xlabel("Iterations (per iteration 1000 inner iterations)")
    plt.ylabel("Error of 0.5*||A - WH||_Fro^2")
    plt.show()


# Print unmixing information
def print_unmixing_data(name, A, W_final, H_final, error_list, total_time_list):
    total_iters = len(error_list)

    print(f"\n\n{name.capitalize()} Dataset:\n")

    print(f"Original Data Cube(reshaped to 2D)(m x n x p -> m*n x p) => {A.shape}:\n")
    print(A, "\n")

    print(f"\nUnmixed Matrices W and H after {total_iters} * 1000 iterations:\n")

    print(f"\nW Dimensions: ({W_final.shape[0]} x {W_final.shape[1]})\n")
    print(W_final, "\n")

    print(f"\nH Dimensions: ({H_final.shape[0]} x {H_final.shape[1]})\n")
    print(H_final, "\n")

    print("Error after each outer iteration: ", error_list)
    print("Total time(in s) for each outer iteration: ", total_time_list)