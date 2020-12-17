import numpy as np
import os
import matplotlib.pyplot as plt
import NMF


# synthetic data
def synthetic_data(m, n, j=100, sigma=0.1):
    X = np.random.random((m, j))
    Y = np.random.random((j, n))
    N = np.random.normal(0, sigma, (m, n))
    A = np.abs(np.dot(X, Y) + N)
    return A

# synthetic data 2


def synthetic_data2(m, n, r=100):
    return np.abs(np.dot(np.random.random((m, r)), np.random.random((r, n))))

# another kind of synthetic data


def synthetic_data2(m, n, k):
    A = np.abs(np.dot(np.random.random((m, k)), np.random.random((k, n))))
    return A


# Average of error and time for (R)NMF decomposition
def av_et_NMF(m=0, n=0, r=0, n_it=0, k=0, A=None, er_out=False):
    er = np.zeros(n_it)
    t_tot = np.zeros(n_it)
    t_rp = np.zeros(n_it)

    for it in range(n_it):
        # print 'it = ', it
        if A is None:
            if m > 0 and n > 0:
                A = synthetic_data(m, n)
            else:
                print("error: A, m and n = 0")
        W, H, er[it], t_tot[it], t_rp[it] = NMF.et_NMF(A, r, k, er_out)
    
    print("W: ", W, "\n")
    print("H: ", H, "\n\n")

    return np.mean(er), np.mean(t_tot), np.mean(t_rp)


# NMF for different values of m
def m_nmf(name, mrange, n, r, n_it, k=0, er_out=False):
    n_m = np.size(mrange)

    error = np.zeros(n_m)
    t_tot = np.zeros(n_m)
    t_rp = np.zeros(n_m)

    for i in range(n_m):
        m = mrange[i]
        print ('m = ', m)
        error[i], t_tot[i], t_rp[i] = av_et_NMF(m, n, r, n_it, k, A=None,
                                                er_out=er_out)

    try:
        os.mkdir('data')
    except OSError:
        pass

    if k == 0:
        np.savez('data/nmf_'+name+'_m.npz',
                 mrange=mrange, error=error, t_tot=t_tot)
    else:
        np.savez('data/rnmf_'+name+'_m.npz', mrange=mrange,
                 error=error, t_tot=t_tot, t_rp=t_rp)


# NMF for different values of n
def n_nmf(name, m, nrange, r, n_it, k=0, er_out=False):
    n_n = np.size(nrange)

    error = np.zeros(n_n)
    t_tot = np.zeros(n_n)
    t_rp = np.zeros(n_n)

    for i in range(n_n):
        n = nrange[i]
        print('n = ', n)
        error[i], t_tot[i], t_rp[i] = av_et_NMF(m, n, r, n_it, k, A=None,
                                                er_out=er_out)

    try:
        os.mkdir('data')
    except OSError:
        pass

    if k == 0:
        np.savez('data/nmf_'+name+'_n.npz',
                 nrange=nrange, error=error, t_tot=t_tot)
    else:
        np.savez('data/rnmf_'+name+'_n.npz', nrange=nrange,
                 error=error, t_tot=t_tot, t_rp=t_rp)

# NMF for different values of r


def r_nmf(name, rrange, n_it, k=0, m=0, n=0, A=None, er_out=False):
    n_r = np.size(rrange)

    error = np.zeros(n_r)
    t_tot = np.zeros(n_r)
    t_rp = np.zeros(n_r)

    for i in range(n_r):
        r = rrange[i]
        print('r = ', r)
        error[i], t_tot[i], t_rp[i] = av_et_NMF(m, n, r, n_it, k, A, er_out)

    try:
        os.mkdir('data')
    except OSError:
        pass

    if k == 0:
        np.savez('data/nmf_'+name+'_r.npz',
                 rrange=rrange, error=error, t_tot=t_tot)
        return error, t_tot
    else:
        np.savez('data/rnmf_'+name+'_r.npz', rrange=rrange,
                 error=error, t_tot=t_tot, t_rp=t_rp)
        return error, t_tot, t_rp

# NMF for different values of k


def k_nmf(name, r, n_it, krange, m=0, n=0, A=None, er_out=False, random_proj=True):
    if random_proj == False:
        error, t_tot, _ = av_et_NMF(m, n, r, n_it, k=0, A=A, er_out=True)
    else:
        n_k = np.size(krange)
        error = np.zeros(n_k)
        t_tot = np.zeros(n_k)
        t_rp = np.zeros(n_k)

        for i in range(n_k):
            k = krange[i]
            print('k = ', k)
        error[i], t_tot[i], t_rp[i] = av_et_NMF(m, n, r, n_it, k, A, er_out)

    try:
        os.mkdir('data')
    except OSError:
        pass

    if random_proj == False:
        np.savez('data/nmf_'+name+'_k.npz', krange=krange,
                 error=np.ones(np.size(krange))*error,
                 t_tot=np.ones(np.size(krange))*t_tot)
        return error, t_tot
    else:
        np.savez('data/rnmf_'+name+'_k.npz', krange=krange,
                 error=error, t_tot=t_tot, t_rp=t_rp)
        return error, t_tot, t_rp


# make three plots
def plot_nmf(range1, range2, error1, t1, error2, t2, t_rp, type1, type2):
    type2_c = type2[0:3]

    try:
        os.mkdir('plot')
    except OSError:
        pass

    # plot error
    plt.figure(1)
    plt.plot(range1, error1, 'r', label=type2_c, linewidth=2, marker='o',
             linestyle='-')
    plt.plot(range2, error2, 'b', label='Randomized '+type2_c, linewidth=2,
             marker='o', linestyle='-')
    plt.ylabel('error', fontsize=10)
    plt.xlabel(type1, fontsize=10)
    plt.legend(loc='best', fontsize=10)
    plt.xlim(min(range1[0], range2[0]), max(range1[-1], range2[-1]))
    plt.ylim(0, max([np.nanmax(error1), np.nanmax(error2)]) * 1.1)
    plt.savefig('plot/'+type2+'_'+type1+'_error.pdf')

    # plot time
    plt.figure(2)
    plt.plot(range1, t1, 'r', label=type2_c, linewidth=2, marker='o',
             linestyle='-')
    plt.plot(range2, t2, 'b', label='Randomized '+type2_c, linewidth=2,
             marker='o', linestyle='-')
    plt.plot(range2, t_rp, 'g', label='RP', linewidth=2, marker='o',
             linestyle='-')
    plt.ylabel('computational time (s)', fontsize=10)
    plt.xlabel(type1, fontsize=10)
    plt.legend(loc='best', fontsize=10)
    plt.xlim(min(range1[0], range2[0]), max(range1[-1], range2[-1]))
    plt.ylim(0, max([np.nanmax(t1), np.nanmax(t2)]) * 1.1)
    plt.savefig('plot/'+type2+'_'+type1+'_time.pdf')

    # plot ratio
    plt.figure(3)
    plt.plot(range2, np.nan_to_num(t_rp/t2), 'b', label='ratio', linewidth=2,
             marker='o', linestyle='-')
    plt.ylabel('ratio', fontsize=10)
    plt.xlabel(type1, fontsize=10)
    plt.legend(loc='best', fontsize=10)
    plt.xlim(range2[0], range2[-1])
    plt.ylim(0, np.nanmax(np.nan_to_num(t_rp/t2)) * 1.1)
    plt.savefig('plot/'+type2+'_'+type1+'_ratio.pdf')

    # plt.show()
    plt.close('all')

# read and plot


def run_plot(type1, type2):
    # read first file
    data1 = np.load('data/'+type2+'_'+type1+'.npz')
    range1 = data1[type1+'range']
    error1 = data1['error']
    t1 = data1['t_tot']

    # read second file
    data2 = np.load('data/r'+type2+'_'+type1+'.npz')
    range2 = data2[type1+'range']
    error2 = data2['error']
    t2 = data2['t_tot']
    t_rp = data2['t_rp']

    # plot
    plot_nmf(range1, range2, error1, t1, error2, t2, t_rp, type1, type2)
