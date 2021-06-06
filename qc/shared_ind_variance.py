"""
For each site, how does the amount of shared variance relate to the amount of independent variance? 
    - show as a fraction of total variance
Idea is that sites where indep. noise dominates (for some reason) might be sites where second order /
decoding effects of pupil might be outliers.
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR2, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, CPN_SITES
import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import numpy as np

zscore = True
sites = CPN_SITES + HIGHR_SITES
batches = [331]*len(CPN_SITES) + [289]*len(HIGHR_SITES)

for site, batch in zip(sites, batches):
    if site in ['BOL005c', 'BOL006b']:
        BATCH = 294
    else:
        BATCH = batch
    X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=BATCH, 
                                       return_epoch_list=True,
                                       exclude_low_fr=False,
                                       threshold=1)

    # remove spont
    X = X[:, :, :, ~sp_bins[0, 0, 0, :]]
    pup_mask =  pup_mask[0, :, :, ~sp_bins[0, 0, 0, :]]

    # get noise (residual)
    Xr = X - X.mean(axis=1, keepdims=True)
    if zscore:
        sd = Xr.std(axis=1, keepdims=True)
        sd[sd==0] = 1
        Xr /= sd

    # reshape for eigendecomposition
    Xr = Xr.reshape(Xr.shape[0], -1)
    pup_mask = pup_mask.reshape(1, -1).squeeze()

    cov = np.cov(Xr)
    evals, evecs = np.linalg.eig(cov)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    sn_fract_var = Xr.var(axis=-1) / Xr.var(axis=-1).sum()

    f, ax = plt.subplots(1, 3, figsize=(9, 3))
    
    ax[0].plot(evals / evals.sum(), 'o-', label='PC')
    ax[0].set_ylabel('Fraction variance explained')
    ax[0].set_xlabel("Principal component / neuron")
    ax2 = ax[0].twinx()
    ax2.spines['right'].set_visible(True)
    ax2.plot(sn_fract_var, 'o-', color='orange', label='Single neuron')
    ax2.set_ylabel(r"Fract. var exp. by single neuron", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(ax[0].get_ylim())

    # plot state-dependent projections of data
    big_pc = Xr[:, pup_mask].T.dot(evecs).T
    small_pc = Xr[:, ~pup_mask].T.dot(evecs).T
    allpc = Xr.T.dot(evecs).T
    big_pc_var = big_pc.var(axis=-1) / allpc.var(axis=-1).sum() #big_pc.var(axis=-1).sum()
    small_pc_var = small_pc.var(axis=-1) / allpc.var(axis=-1).sum() #small_pc.var(axis=-1).sum()

    ax[1].plot(big_pc_var, '.-', label='Large', color='red')
    ax[1].plot(small_pc_var, '.-', label='Small', color='blue')
    ax[1].legend(frameon=False)

    # single neurons
    big_sn = Xr[:, pup_mask].var(axis=-1) / allpc.var(axis=-1).sum() #Xr[:, pup_mask].var(axis=-1).sum()
    small_sn = Xr[:, ~pup_mask].var(axis=-1) / allpc.var(axis=-1).sum() #Xr[:, ~pup_mask].var(axis=-1).sum()
    ax[1].plot(big_sn, '-', color='red')
    ax[1].plot(small_sn, '-', color='blue')

    ax[2].imshow(evecs.T, cmap='bwr', vmin=-1, vmax=1)
    ax[2].set_ylabel('Component')
    ax[2].set_xlabel('Neuron')
    ax[2].set_title("Loading vector")

    f.suptitle(site)
    f.tight_layout()
    
    plt.show()

    # TODO - show pupil dependent single neuron and pupil-dependent shared variance in this same format