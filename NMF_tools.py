import numpy as np
import os
import matplotlib.pyplot as plt
import NMF

# Average of error and time for NMF decomposition
def _avg_err_time_NMF(r, n_it, A=None, W_prev=None, H_prev=None, er_out=False):
    er = np.zeros(n_it)
    t_tot = np.zeros(n_it)
    # t_rp = np.zeros(n_it)
    W_list = [None] * n_it
    H_list = [None] * n_it

    for it in range(n_it):
        print('it = ', it)
        if A is None:
            print("Error: A is None")
        else:
            W_list[it], H_list[it], er[it], t_tot[it] = NMF.et_NMF(A, r, er_out, W_prev=W_prev, H_prev=H_prev)
            
            # print("W: ", W_list[it], "\n")
            # print("H: ", H_list[it], "\n")
            print("Error: ", er[it], "\n\n")

    return W_list, H_list, er, t_tot


# NMF for different values of r
def r_nmf(name, r, n_it, A=None, W_prev=None, H_prev=None, er_out=False):
    print('r = ', r)
    W_list, H_list, error, t_tot = _avg_err_time_NMF(r, n_it, A, W_prev, H_prev, er_out)

    try:
        os.mkdir('data')
    except OSError:
        pass

    np.savez('data/nmf_'+name+'_W_H.npz', W_list=W_list, H_list=H_list)

    np.savez('data/nmf_'+name+'_r.npz', r=r, error=error, t_tot=t_tot)
    return W_list, H_list, error, t_tot
