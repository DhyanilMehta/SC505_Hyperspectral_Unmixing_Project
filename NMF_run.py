import os
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import NMF_tools as nmft

def real_dataset(name):
    if name == "paviaU":
        mat = scipy.io.loadmat('dataset/PaviaU.mat')
    else:
        mat = scipy.io.loadmat('dataset/'+name.capitalize()+'.mat')
    
    data = np.asarray_chkfinite(mat[name])
    data_matrix = np.abs(data.reshape(data.shape[0] * data.shape[1], data.shape[2]))
    return data_matrix

ip_endmember_names = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Tower"]

slsA_endmember_names = ["Brocoli_green_weeds_1", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk"]

paviaU_endmember_names = ["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"]

# name = 'salinasA_corrected'
# endmembers = 6
# scale_factor = 100
# endmember_names = slsA_endmember_names

name = 'indian_pines_corrected'
endmembers = 16
scale_factor = 500
endmember_names = ip_endmember_names

# name = 'paviaU'
# endmembers = 9
# scale_factor = 500
# endmember_names = paviaU_endmember_names

# Extract hyperspectral data cube
A = real_dataset(name)
A = A / scale_factor

n_it = 0
error_list = [np.inf]
W_list = [None]
H_list = [None]
total_time_list = []

while True:
    n_it += 1
    W_it, H_it, err, total_time = nmft.calc_NMF(A, endmembers, W_prev=W_list[n_it - 1], H_prev=H_list[n_it - 1])
    
    print("Iteration: ", n_it, "Error: ", err)
    W_list.append(W_it)
    H_list.append(H_it)
    error_list.append(err)
    total_time_list.append(total_time)

    if math.isclose(error_list[n_it - 1], error_list[n_it], rel_tol=0.001, abs_tol=0.01):
        break

error_list.pop(0)
W_list.pop(0)
H_list.pop(0)

try:
    os.mkdir('data')
except OSError:
    pass

np.savez('data/nmf_'+name+'_data.npz', W_list=W_list, H_list=H_list, endmember_names=endmember_names, error=error_list, total_time=total_time_list)

W_final = W_list.pop()
H_final = H_list.pop()


# Print unmixed information
nmft.print_unmixing_data(name, A, W_final, H_final, error_list, total_time_list)

# Plot graphs
nmft.plot_NMF_data(name, endmember_names)