import numpy as np
import sys
# sys.path.insert(0, '../core/')
import NMF_tools as nmft
import scipy.io

def real_dataset(name):
    mat = scipy.io.loadmat('dataset/'+name.capitalize()+'.mat')
    data = np.asarray_chkfinite(mat[name])
    data_matrix = np.abs(data.reshape(np.shape(data)[0] * np.shape(data)[1],
                                      np.shape(data)[2]))
    return data_matrix

er_out = True

# name = 'salinas'
# A = real_dataset(name)

# rrange = np.arange(1, 20, 2)
# k = 500
# salinas_endmembers = 32
# n_it = 3
# nmft.r_nmf(name=name, r=salinas_endmembers, n_it=n_it, k=0, A=A, er_out=er_out)
# nmft.r_nmf(name=name, rrange=rrange, n_it=n_it, k=k, A=A, er_out=er_out)

# r = 10
# krange = np.asarray([10, 20, 50, 100, 200, 500, 1000])
# n_it = 3
# nmft.k_nmf(name=name, r=r, n_it=n_it, krange=krange, A=A, er_out=er_out,
#        random_proj=False)
# nmft.k_nmf(name=name, r=r, n_it=n_it, krange=krange, A=A, er_out=er_out)

# nmft.run_plot('k', 'nmf_'+name)
# nmft.run_plot('r', 'nmf_'+name)


name = 'indian_pines'
A = real_dataset(name)

# rrange = np.arange(1, 20, 2)
# k = 500
indian_pines_endmembers = 16
n_it = 10
W_list, H_list, error, total_time = nmft.r_nmf(name=name, r=indian_pines_endmembers, n_it=n_it, k=0, A=A, er_out=er_out)
# nmft.r_nmf(name=name, rrange=rrange, n_it=n_it, k=k, A=A, er_out=er_out)

# r = 10
# krange = np.asarray([10, 20, 50, 100, 200, 500, 1000])
# n_it = 3
# # nmft.k_nmf(name=name, r=r, n_it=n_it, krange=krange, A=A, er_out=er_out,
# #        random_proj=False)
# # nmft.k_nmf(name=name, r=r, n_it=n_it, krange=krange, A=A, er_out=er_out)

# nmft.run_plot('k', 'nmf_'+name)
# nmft.run_plot('r', 'nmf_'+name)

print("Indian Pines: \n\n")

print("Original Data Cube(reshaped to 2D)(m x n x p -> m*n x p):\n")
print(np.shape(A), "\n")
print(A, "\n")

print(f"\nUnmixed Matrices W and H for {n_it} outer iterations:\n")
for it in range(n_it):
    print(f"\nW_{it} Dimensions: ({np.shape(W_list[it])[0]} x {np.shape(W_list[it])[1]})\n")
    print(W_list[it], "\n")

    print(f"\nH_{it} Dimensions: ({np.shape(H_list[it])[0]} x {np.shape(H_list[it])[1]})\n")
    print(H_list[it], "\n")

print("Error for each outer iteration: ", error.tolist())
print("Total time(in s) for each outer iteration: ", total_time.tolist())