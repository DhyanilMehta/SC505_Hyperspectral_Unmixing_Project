import numpy as np
import sys
sys.path.insert(0, '../core/')
import NMF_tools as nmft

name = 'synth'
er_out = True

mrange  = np.asarray([700, 1000, 1300, 1700, 2000, 2500, 3000])
n       = 2000
r       = 10
k       = 500
n_it    = 3
nmft.m_nmf(name=name, mrange=mrange, n=n, r=r, n_it=n_it, k=0, er_out=er_out)
nmft.m_nmf(name=name, mrange=mrange, n=n, r=r, n_it=n_it, k=k, er_out=er_out)

m       = 2000
nrange  = np.asarray([700, 1000, 1300, 1700, 2000, 2500, 3000])
r       = 10
k       = 500
n_it    = 3
nmft.n_nmf(name=name, m=m, nrange=nrange, r=r, n_it=n_it, k=0, er_out=er_out)
nmft.n_nmf(name=name, m=m, nrange=nrange, r=r, n_it=n_it, k=k, er_out=er_out)

m       = 400
n       = 400
rrange  = np.arange(10, 40, 4)
k       = 500
n_it    = 3
#nmft.r_nmf(name=name, rrange=rrange, n_it=n_it, k=0, m=m, n=n, er_out=er_out)
#nmft.r_nmf(name=name, rrange=rrange, n_it=n_it, k=k, m=m, n=n, er_out=er_out)

m       = 400
n       = 400
r       = 10
krange  = np.arange(10, 200, 25)
n_it    = 3
#nmft.k_nmf(name=name, r=r, n_it=n_it, krange=krange, m=m, n=n, er_out=er_out,
#        random_proj=False)
#nmft.k_nmf(name=name, r=r, n_it=n_it, krange=krange, m=m, n=n, er_out=er_out)


#nmft.run_plot('m', 'nmf_'+name)
#nmft.run_plot('n', 'nmf_'+name)
#nmft.run_plot('r', 'nmf_'+name)
#nmft.run_plot('k', 'nmf_'+name)