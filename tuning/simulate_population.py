"""
Simulate a population, determine if delta d-prime changes 
are predictable.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from charlieTools.nat_sounds_ms import decoding
import charlieTools.plotting as cplt

nTrials = 20
site = 'DRX008b.e65:128'
batch = 289

# load a sample recording to get mean responsesX, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)
X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)
ncells = X.shape[0]
nreps = X.shape[1]
nstim = X.shape[2]
nbins = X.shape[3]
sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
nstim = nstim * nbins
spont_bins = np.argwhere(sp_bins[0, 0, :])
X = X.reshape(ncells, nreps, nstim)
Xpup = X_pup.reshape(1, nreps, nstim)
ev_bins = list(set(range(X.shape[-1])).difference(set(spont_bins.squeeze())))
Xev = X[:, :, ev_bins]
Xpup = Xpup[:, :, ev_bins]

# compute mean response, add noise
Xu = Xev.mean(axis=1)
nstim = Xev.shape[-1]

pca = PCA()
pca.fit(Xu.T)

# additive noise, one dim, depends on pupil (give pupil the same mean / sem as the true data)
# biased to be positive
rv = np.random.normal(1, 1, (1, ncells))
rv /= np.linalg.norm(rv)
rv = pca.components_[[0], :]

# first order loading (~orthogonal to noise)


# baseline covariance matrix
base_cov = rv.T.dot(rv)

# iteratively sample single trials (so that you get a dependence on pupil this way)
X = np.zeros((ncells, nTrials, nstim))
Xbp = np.zeros((ncells, int(nTrials / 2), nstim))
Xsp = np.zeros((ncells, int(nTrials / 2), nstim))
p = np.zeros((1, nTrials, nstim))
pm = np.zeros((1, nTrials, nstim))
pmax = Xpup.max()
for s in range(nstim):
    bpi = 0
    spi = 0
    pupil = np.random.normal(Xpup[0, :, s].mean(), Xpup[0, :, s].std(), nTrials) / pmax
    p[0, :, s] = pupil
    pm[0, :, s] = pupil >= np.median(pupil)
    for t in range(nTrials):
        d = Xev[:, :, s].var(axis=1)
        #d *= pupil[t]
        cov = base_cov * (1 / pupil[t])
        np.fill_diagonal(cov, d)
        X[:, t, s] = np.random.multivariate_normal(Xu[:, s] * (pupil[t] + 0.5), cov=cov, size=(1,))
        if pm[0, t, s]:
            Xbp[:, bpi, s] = X[:, t, s]
            bpi += 1
        else:
            Xsp[:, spi, s] = X[:, t, s]
            spi += 1


pca = PCA()
pca.fit(X.mean(axis=1).T)

pax = pca.components_[0:2, :].T
#pax = np.concatenate((pca.components_[[0], :], rv), axis=0).T
proj = X.T.dot(pax)
bproj = Xbp.T.dot(pax)
sproj = Xsp.T.dot(pax)

f, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
for i in range(0, proj.shape[0]):
    el = cplt.compute_ellipse(proj[i, :, 0], proj[i, :, 1])
    ax[0].plot(el[0], el[1], color='grey', lw=0.5)
    el = cplt.compute_ellipse(bproj[i, :, 0], bproj[i, :, 1])
    ax[1].plot(el[0], el[1], color='grey', lw=0.5)
    el = cplt.compute_ellipse(sproj[i, :, 0], sproj[i, :, 1])
    ax[2].plot(el[0], el[1], color='grey', lw=0.5)

ax[0].axis('square')
for a in ax.flatten():
    a.axvline(0, linestyle='--', color='k', zorder=-1, lw=1)
    a.axhline(0, linestyle='--', color='k', zorder=-1, lw=1)
ax[0].set_title("All data")
ax[1].set_title("Big pupil")
ax[2].set_title("Small pupil")

f.tight_layout()

plt.show()