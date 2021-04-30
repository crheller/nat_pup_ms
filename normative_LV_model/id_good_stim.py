"""
For whatever reason, some stimuli might drive very weak responses and have unreliable noise distributions. 
We don't want to try to fit out latent variable on these unreliable stimuli. So, for each site, identify the best
X number of stimuli for the model to be fit on based on how consistently we can estimate their first PC
"""
from global_settings import HIGHR_SITES

import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import os
import pandas as pd 
import pickle
import sys

import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.dim_reduction as dr

import nems
import nems_lbhb.baphy as nb
import nems.db as nd
import logging

log = logging.getLogger(__name__)

sorted_epochs = {}
path = '/auto/users/hellerc/results/nat_pupil_ms/reliable_epochs/'
for site in HIGHR_SITES:
    if site in ['BOL005c', 'BOL006b']:
        batch = 294
    else:
        batch = 289

    X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)
    ncells = X.shape[0]
    nreps_raw = X.shape[1]
    nstim = X.shape[2]
    nbins = X.shape[3]
    sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
    nstim = nstim * nbins

    # z-score
    _, X = nat_preproc.scale_est_val(X, X)

    # =========================== get list of evoked epoch / bin combos ==========================
    spont_bins = np.argwhere(sp_bins[0, 0, :])

    # get list of epoch/bins as a tuple 
    epochs_bins = np.concatenate([[(e, k, i) for k in range(nbins)] for i, e in enumerate(epochs)])
    # remove spont
    epochs_bins = [tuple(epochs_bins[i]) for i in range(len(epochs_bins)) if i not in spont_bins]

    # calculate uncertainty of 1st PC across random samples of 50% of the data
    nsamples = 10
    evec_err = []
    for ebi in epochs_bins:
        d = X[:, :, int(ebi[-1]), int(ebi[1])]
        ss = int(d.shape[1] / 2)
        vecs = []
        for n in range(nsamples):
            sidx = np.random.choice(np.arange(0, d.shape[1]), ss, replace=False)
            sdata = d[:, sidx]
            # get first PC
            evals, evecs = np.linalg.eig(np.cov(sdata))
            evecs = np.real(evecs[:, np.argsort(evals)[::-1]])
            vecs.append(evecs[:, 0])
        
        sem = np.std(np.stack(vecs), axis=0) / np.sqrt(nsamples)
        err = np.sqrt(np.sum(sem**2))
        evec_err.append(err)

    # generate a null distribution
    null_err = []
    for i in range(1000):
        rvecs = []
        for j in range(nsamples):
            vec = np.random.normal(0, 1, size=(X.shape[0],1))
            vec /= np.linalg.norm(vec)
            rvecs.append(vec)
        sem = np.std(np.stack(rvecs), axis=0) / np.sqrt(nsamples)
        err = np.sqrt(np.sum(sem**2))
        null_err.append(err)

    # plot the two
    bins = np.arange(np.min(null_err+evec_err), np.max(null_err+evec_err), 0.005)
    f, ax = plt.subplots(1, 1)

    ax.hist(evec_err, bins=bins, density=True, histtype='step')
    ax.hist(null_err, bins=bins, density=True, histtype='step')

    f.canvas.set_window_title(site)

    # save the epochs sorted according to their reliability
    # also add a mask over those that are more reliable than null
    ep_bin = [(eb[0], int(eb[1])) for eb in epochs_bins]
    sidx = np.argsort(evec_err)
    thissite = {}
    thissite['sorted_epochs'] = [tuple(x) for x in np.array(ep_bin)[sidx]]
    thissite['reliable_mask'] = (np.array(evec_err)< np.array(null_err).min())[sidx]
    sorted_epochs[site] = thissite
    # save for this site
    pickle.dump(thissite, open(f"{path}{site}.pickle", "wb"))

plt.show()