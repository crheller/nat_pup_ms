"""
Develop a function to load FA model and generate 1 of 4 possible simulated datasets:
    * change in gain only
    * change in indep only (fixing absolute covariance)
    * change in indep only (fixing relative covariance - so off-diagonals change)
    * change in everything (full FA simulation)
"""
import charlieTools.nat_sounds_ms.decoding as decoding
import matplotlib.pyplot as plt
import numpy as np
import pickle


site = "AMT020a"
batch = 331
X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)
ncells = X.shape[0]
nreps_raw = X.shape[1]
nstim = X.shape[2]
nbins = X.shape[3]
sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
nstim = nstim * nbins

big_psth = np.stack([X.reshape(ncells, nreps_raw, nstim)[:, pup_mask.reshape(1, nreps_raw, nstim)[0, :, s], s].mean(axis=1) for s in range(nstim)]).T.reshape(ncells, X.shape[2], nbins)[:, np.newaxis, :, :]
small_psth = np.stack([X.reshape(ncells, nreps_raw, nstim)[:, pup_mask.reshape(1, nreps_raw, nstim)[0, :, s]==False, s].mean(axis=1) for s in range(nstim)]).T.reshape(ncells, X.shape[2], nbins)[:, np.newaxis, :, :]

Xsim1, psim1 = decoding.load_FA_model(site, batch, big_psth, small_psth, sim=1, nreps=10000)
Xsim2, psim2 = decoding.load_FA_model(site, batch, big_psth, small_psth, sim=2, nreps=10000)
Xsim3, psim3 = decoding.load_FA_model(site, batch, big_psth, small_psth, sim=3, nreps=10000)
Xsim4, psim4 = decoding.load_FA_model(site, batch, big_psth, small_psth, sim=4, nreps=10000)
Xsim5, psim5 = decoding.load_FA_model(site, batch, big_psth, small_psth, sim=5, nreps=10000)

pair = (27, 30)
stim = (1, 2)

f, ax = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)

for i, Xs in zip(range(4), [Xsim1, Xsim2, Xsim3, Xsim4]):
    xb = Xs[pair[0], psim1[0, :, stim[0], stim[1]], stim[0], stim[1]]
    yb = Xs[pair[1], psim1[0, :, stim[0], stim[1]], stim[0], stim[1]]
    xs = Xs[pair[0], psim1[0, :, stim[0], stim[1]]==False, stim[0], stim[1]]
    ys = Xs[pair[1], psim1[0, :, stim[0], stim[1]]==False, stim[0], stim[1]]
    ax[0, i].scatter(xb, yb, s=1, alpha=0.5)
    cc = np.round(np.corrcoef(xb, yb)[0,1], 3)
    cov = np.round(np.cov(xb, yb)[0,1], 3)
    v1 = np.round(np.var(xb), 3)
    v2 = np.round(np.var(yb), 3)
    ax[0, i].set_title(f"cc: {cc} \n cov: {cov} \n var1: {v1}, var2: {v2}")
    ax[1, i].scatter(xs, ys, s=1, alpha=0.5, color="tab:orange")
    cc = np.round(np.corrcoef(xs, ys)[0,1], 3)
    cov = np.round(np.cov(xs, ys)[0,1], 3)
    v1 = np.round(np.var(xs), 3)
    v2 = np.round(np.var(ys), 3)
    ax[1, i].set_title(f"cc: {cc} \n cov: {cov} \n var1: {v1}, var2: {v2}")

f.tight_layout()

# evaluate over all neurons given stimulus
f, ax = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
bb = np.var(Xsim1[:, psim1[0, :, 0, 0]==True, stim[0], stim[1]], axis=1)
s1 = np.var(Xsim1[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]], axis=1)
ax[0].scatter(bb, s1, s=25)
s2 = np.var(Xsim2[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]], axis=1)
ax[1].scatter(bb, s2, s=25)
s3 = np.var(Xsim3[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]], axis=1)
ax[2].scatter(bb, s3, s=25)
s4 = np.var(Xsim4[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]], axis=1)
ax[3].scatter(bb, s4, s=25)
s5 = np.var(Xsim5[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]], axis=1)
bb = np.var(Xsim5[:, psim1[0, :, 0, 0]==True, stim[0], stim[1]], axis=1)
ax[4].scatter(bb, s5, s=25)
for a in ax:
    a.plot([0, 15], [0, 15], "k--")
f.suptitle("Single neuron variance")

# evaluate over all neurons given stimulus cc
f, ax = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
bb = np.corrcoef(Xsim1[:, psim1[0, :, 0, 0]==True, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
s1 = np.corrcoef(Xsim1[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[0].scatter(bb, s1, s=5)
s2 = np.corrcoef(Xsim2[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[1].scatter(bb, s2, s=5)
s3 = np.corrcoef(Xsim3[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[2].scatter(bb, s3, s=5)
s4 = np.corrcoef(Xsim4[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[3].scatter(bb, s4, s=5)
s5 = np.corrcoef(Xsim5[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
bb = np.corrcoef(Xsim5[:, psim1[0, :, 0, 0]==True, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[4].scatter(bb, s5, s=5)
for a in ax:
    a.plot([-.2, 0.5], [-.2, 0.5], "k--")
    a.set_xlim((-.2, 0.5))
    a.set_ylim((-.2, 0.5))
f.suptitle("correlation coefficient")


f, ax = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
bb = np.cov(Xsim1[:, psim1[0, :, 0, 0]==True, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
s1 = np.cov(Xsim1[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[0].scatter(bb, s1, s=5)
s2 = np.cov(Xsim2[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[1].scatter(bb, s2, s=5)
s3 = np.cov(Xsim3[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[2].scatter(bb, s3, s=5)
s4 = np.cov(Xsim4[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[3].scatter(bb, s4, s=5)
s5 = np.cov(Xsim5[:, psim1[0, :, 0, 0]==False, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
bb = np.cov(Xsim5[:, psim1[0, :, 0, 0]==True, stim[0], stim[1]])[np.triu_indices(X.shape[0], 1)]
ax[4].scatter(bb, s5, s=5)
for a in ax:
    a.plot([-1, 2], [-1, 2], "k--")
    a.set_xlim((-1, 2))
    a.set_ylim((-1, 2))
f.suptitle("raw covariance")