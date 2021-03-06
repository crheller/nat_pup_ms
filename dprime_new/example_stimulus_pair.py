"""
Reload dataset. Visualize pairwise stimulus discrimination(s).
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cross_decomposition import PLSRegression

import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr

#np.random.seed(123)

zscore = True
regress_pupil = False
el_only = False
sim1 = False
sim2 = False
njacks = 10 # generate one random est / val set
jk_set = 2 # which jack set to use for plotting 
combo = None #(56, 58)  # if None, just using first evoked/evoked combo
site = 'TAR010c'
batch = 289
site = 'AMT020a'
batch = 331

# ================================= load recording ==================================
X, sp_bins, X_pup, pup_mask = decoding.load_site(site=site, batch=batch, 
                                                                        sim_first_order=sim1,
                                                                        sim_second_order=sim2,
                                                                        regress_pupil=regress_pupil,
                                                                        verbose=False)
ncells = X.shape[0]
nreps = X.shape[1]
nstim = X.shape[2]
nbins = X.shape[3]
X = X.reshape(ncells, nreps, nstim * nbins)
sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)

# mask pupil per stimulus, rather than overall
pup_mask = pup_mask.reshape(1, nreps, nstim * nbins)
nstim = nstim * nbins


# =========================== generate list of est/val sets ==========================
est, val, p_est, p_val = nat_preproc.get_est_val_sets(X, pup_mask=pup_mask, njacks=njacks)
nreps_train = est[0].shape[1]
nreps_test = val[0].shape[1]

# determine number of dim reduction components (bounded by ndim in dataset) 
components = np.min([ncells, nreps_train])

# ============================ preprocess est / val sets =============================
if zscore:
    est, val = nat_preproc.scale_est_val(est, val)
else:
    # just center data
    est, val = nat_preproc.scale_est_val(est, val, sd=False)

# =========================== generate a list of stim pairs ==========================
all_combos = list(combinations(range(nstim), 2))
spont_bins = np.argwhere(sp_bins[0, 0, :])
spont_combos = [c for c in all_combos if (c[0] in spont_bins) & (c[1] in spont_bins)]
ev_ev_combos = [c for c in all_combos if (c[0] not in spont_bins) & (c[1] not in spont_bins)]
spont_ev_combos = [c for c in all_combos if (c not in ev_ev_combos) & (c not in spont_combos)]

if combo is None:
    combo = ev_ev_combos[0]

# Define xtrain/test, do dim reduction, project stim A and stim B
est = est[jk_set][:, :, [combo[0], combo[1]]]
xtrain = nat_preproc.flatten_X(est[:, :, :, np.newaxis])
val = val[jk_set][:, :, [combo[0], combo[1]]]
xtest = nat_preproc.flatten_X(val[:, :, :, np.newaxis])

p_est = p_est[jk_set][:, :, [combo[0], combo[1]]]
p_val = p_val[jk_set][:, :, [combo[0], combo[1]]]

Y = dr.get_one_hot_matrix(ncategories=2, nreps=nreps_train)
#pls = PLSRegression(n_components=2, max_iter=500, tol=1e-7)
#pls.fit(xtrain.T, Y.T)
#pls_weights = pls.x_weights_
tdr = dr.TDR()
tdr.fit(xtrain.T, Y.T)
tdr_weights = tdr.weights

xtrain = (xtrain.T @ tdr_weights.T).T
xtest = (xtest.T @ tdr_weights.T).T

f = decoding.plot_pair(xtrain, xtest, 
                       nreps_train=nreps_train, 
                       nreps_test=nreps_test,
                       train_pmask=p_est,
                       test_pmask=p_val,
                       el_only=el_only)


plt.show()