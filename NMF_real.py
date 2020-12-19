import numpy as np
import sys
import math
# sys.path.insert(0, '../core/')
import NMF_tools as nmft
import scipy.io
import matplotlib.pyplot as plt

def real_dataset(name):
    mat = scipy.io.loadmat('dataset/'+name.capitalize()+'_corrected.mat')
    data = np.asarray_chkfinite(mat[name + '_corrected'])
    data_matrix = np.abs(data.reshape(data.shape[0] * data.shape[1], data.shape[2]))
    return data_matrix

ip_endmember_names = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Tower"]

slsA_endmember_names = ["Brocoli_green_weeds_1", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk"]

er_out = True

name = 'salinasA'
slsA_endmembers = 6

# name = 'indian_pines'
# ip_endmembers = 16


A = real_dataset(name)
A = A / 500
n_it = 1
error = [np.inf]
err_prev = 0
W_list = [None]
H_list = [None]

while not math.isclose(error[0], err_prev, rel_tol=0.00001, abs_tol=0.00001):
    err_prev = error[0]
    W_list, H_list, error, total_time = nmft.r_nmf(name=name, r=slsA_endmembers, n_it=n_it, A=A, W_prev=W_list[0], H_prev=H_list[0], er_out=er_out)


print(f"{name}: \n\n")

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


# Testing purposes
data_WH = np.load(f"data/nmf_{name}_W_H.npz")
# data_WH = np.load("data/nmf_salinas_W_H.npz")
W_l = data_WH["W_list"]
H_l = data_WH["H_list"]

# W_l = [(W + 1000) * 500 for W in H_l]
# H_l = [(H + 1000) * 500 for H in H_l]
# indian_pines_wavelengths = np.linspace(0.4, 2.5, num=A.shape[1])
bands = np.arange(1, A.shape[1] + 1, 1).tolist()

if n_it == 1:
    plt.figure()
    for i in range(H_l[0].shape[0]):
        plt.plot(bands, H_l[0][i])
        plt.legend(slsA_endmember_names, loc="best")
else:
    fig, axs = plt.subplots(1, len(H_l))
    for it in range(len(H_l)):
        for i in range(H_l[it].shape[0]):
            axs[it].plot(bands, H_l[it][i])
# plt.show()

data_r = np.load(f"data/nmf_{name}_r.npz")
# data_r = np.load("data/nmf_salinas_r.npz")
err = data_r["error"]
if n_it != 1:
    iters = np.arange(0, n_it, 1).tolist()
    plt.figure()
    plt.plot(iters, err.tolist())
plt.show()