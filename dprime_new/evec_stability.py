"""
Get a sense of how stable eigenvector measurements are.
If they're very stable, the mean eigenvectors should form orthonormal basis, and
standard error of evals and evecs should be small. Just look at up to top 5 eigenvectors?
"""

import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

loader = decoding.DecodingResults()
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
modelname = 'dprime_jk10'
idx = pd.IndexSlice

# list of sites with > 10 reps of each stimulus
sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']

site = 'BOL006b'

fn = os.path.join(path, site, modelname+'_PLS.pickle')
results = loader.load_results(fn)

ndim = 5
f, ax = plt.subplots(2, ndim-1, figsize=(4 * (ndim-1), 8))
for i, d in enumerate(np.arange(2, ndim+1)):
    dots = []
    du_ev = []
    for combo in results.evoked_stimulus_pairs:
        # get eigenvectors
        evecs = decoding.unit_vectors(results.get_result('evecs_train', combo, d)[0])
        dots.append(abs(evecs.T.dot(evecs)))

        # get overlap of dU and each eigenvector, normalized by the mean eigenvalue
        evals = results.get_result('evals_train', combo, d)[0]
        dU = (results.get_result('dU_train', combo, d)[0].T)
        overlap = (dU.T.dot(evecs) ** 2) / evals
        du_ev.append(overlap)
    
    dot_products = np.mean(np.stack(dots), axis=0)
    snr = np.mean(np.stack(du_ev), axis=0).squeeze()
    snr_sem = np.std(np.stack(du_ev), axis=0).squeeze() / np.sqrt(len(du_ev))

    # plot matrix of pairwise dot products between eigenvectors
    im = ax[0, i].imshow(dot_products, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    ax[0, i].set_title(r'$\mathbf{e}_{\alpha}^{T} \mathbf{e}_{\alpha}$')
    ax[0, i].set_xticks(range(d))
    ax[0, i].set_xticklabels(np.arange(d)+1)
    ax[0, i].set_xlabel(r"$\alpha$")
    ax[0, i].set_yticks(range(d))
    ax[0, i].set_yticklabels(np.arange(d)+1)
    ax[0, i].set_ylabel(r"$\alpha$")

    ax[1, i].errorbar(np.arange(1, d+1), snr, marker='o', linestyle='none', yerr=snr_sem, color='k')
    ax[1, i].set_xlabel(r'$\alpha$')
    ax[1, i].set_ylabel(r'$\frac{(\Delta \mathbf{\mu} \cdot \mathbf{e}_{\alpha})^{2}}{\lambda_{\alpha}}$', fontsize=16)

    

f.tight_layout()
cax,kw = mpl.colorbar.make_axes([a for a in ax[0].flat])
f.colorbar(im, cax=cax, **kw)

plt.show()