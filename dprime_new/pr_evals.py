"""
Compare eigenvalues of raw data to eigenvalues of pupil regressed data.
Point is to show that removing first order pupil decreases magnitude of first PC, thus, 
gain change is mostly along first PC. Which means that first order improvements should
happen most when first PC is aligned with dU. Second order improvements should happen
when second PC is aligned with dU.
"""

import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore'
pr = 'dprime_pr_jk10_zscore'
site = 'DRX006b.e1:64'
n_components = 2

sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']

raw_lambda = []
pr_lambda = []
raw_bp = []
raw_sp = []
pr_bp = []
pr_sp = []
bp_var1 = []
bp_var2 = []
sp_var1 = []
sp_var2 = []
for site in sites:

    fn = os.path.join(path, site, modelname+'_PLS.pickle')
    results = loader.load_results(fn)
    fn = os.path.join(path, site, pr+'_PLS.pickle')
    pr_results = loader.load_results(fn)

    pairs = pr_results.evoked_stimulus_pairs

    raw = results.get_result('evals_test', pairs, n_components)
    pr_vals = pr_results.get_result('evals_test', pairs, n_components)

    '''
    f, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.plot(raw[0].mean(), marker='o', label='raw data')
    ax.plot(pr_vals[0].mean(), marker='o', label='pupil regressed')

    ax.set_ylabel(r"$\lambda$")
    ax.set_xlabel(r"$\alpha$")
    ax.legend(frameon=False)

    f.tight_layout()

    f.canvas.set_window_title(site)
    '''

    raw_lambda.append(raw[0].mean())
    pr_lambda.append(pr_vals[0].mean())

    #bp = results.get_result('bp_evals', pairs, n_components)
    #sp = results.get_result('sp_evals', pairs, n_components)
    #raw_bp.append(bp[0].mean())
    #raw_sp.append(sp[0].mean())

    bp1 = results.slice_array_results('bp_evals', pairs, n_components, idx=[0])[0]
    bp2 = results.slice_array_results('bp_evals', pairs, n_components, idx=[1])[0]
    sp1 = results.slice_array_results('sp_evals', pairs, n_components, idx=[0])[0]
    sp2 = results.slice_array_results('sp_evals', pairs, n_components, idx=[1])[0]
    bp_var1.append((bp1 / (bp1 + bp2)).mean())
    bp_var2.append((bp2 / (bp1 + bp2)).mean())
    sp_var1.append((sp1 / (sp1 + sp2)).mean())
    sp_var2.append((sp2 / (sp1 + sp2)).mean())


    bp = pr_results.get_result('bp_evals', pairs, n_components)
    sp = pr_results.get_result('sp_evals', pairs, n_components)
    pr_bp.append(bp[0].mean())
    pr_sp.append(sp[0].mean())

f, ax = plt.subplots(1, 1, figsize=(4, 4))

delta = np.stack(raw_lambda) - np.stack(pr_lambda)
ax.scatter(np.zeros(len(raw_lambda)) + np.random.normal(0, 0.05, len(raw_lambda)), delta[:,0], 
                    marker='o', color='k', edgecolor='white')
ax.scatter(np.ones(len(raw_lambda)) + np.random.normal(0, 0.05, len(raw_lambda)), delta[:,1], 
                    marker='o', color='k', edgecolor='white')

ax.set_xticks([0, 1])
ax.set_xticklabels([1, 2])

ax.axhline(0, linestyle='--', color='grey')
ax.set_ylabel(r"$\lambda_{\alpha, raw} - \lambda_{\alpha, pr}$")
ax.set_xlabel(r"$\alpha$")

f.tight_layout()

# plot relative SNR on each eigenvector for large / small pupil
f, ax = plt.subplots(1, 1, figsize=(4, 4))

#sp = np.stack(raw_sp).squeeze()[:,1] /  (np.stack(raw_sp).squeeze()[:,0] +  np.stack(raw_sp).squeeze()[:,1]) - \
#         np.stack(raw_bp).squeeze()[:,1] /  (np.stack(raw_bp).squeeze()[:,0] +  np.stack(raw_bp).squeeze()[:,1])
#bp = np.stack(raw_sp).squeeze()[:,0] /  (np.stack(raw_sp).squeeze()[:,0] +  np.stack(raw_sp).squeeze()[:,1]) - \
#            np.stack(raw_bp).squeeze()[:,0] /  (np.stack(raw_bp).squeeze()[:,0] +  np.stack(raw_bp).squeeze()[:,1])

bp1 = bp_var1
sp1 = sp_var1

bp2 = bp_var2
sp2 = sp_var2

ax.scatter(sp1, bp1, marker='o', color='grey', edgecolor='white', s=75, label=r"$\alpha=1$")
ax.scatter(sp2, bp2, marker='o', color='r', edgecolor='white', s=75, label=r"$\alpha=2$")
ax.plot([0.2, 0.8], [0.2, 0.8], 'k--')

ax.set_ylabel(r"$\frac{\lambda_{\alpha}}{\lambda_{1} + \lambda_{2}}$, Large pupil", fontsize=12)
ax.set_xlabel(r"$\frac{\lambda_{\alpha}}{\lambda_{1} + \lambda_{2}}$, Small pupil", fontsize=12)
ax.legend()

f.tight_layout()

plt.show()