import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.baphy as nb
from sklearn.cross_decomposition import PLSRegression
from sklearn .decomposition import PCA
from sklearn.preprocessing import scale
from charlieTools.plotting import compute_ellipse

# load sample recording
site = 'TAR010c'
batch = 289
fs = 4
options = {'cellid': site, 'rasterfs': fs, 'batch': batch, 'pupil': True, 'stim': False}
rec = nb.baphy_load_recording_file(**options)
rec['resp'] = rec['resp'].rasterize()

# add mask for removing pre/post stim silence
rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)

# extract folded matrices and build single response matrix, where each bin is a "stimulus"
epochs = [epoch for epoch in rec.epochs.name.unique() if 'STIM_00' in epoch]
rec = rec.and_mask(epochs)

resp_dict =rec['resp'].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)
for i, epoch in enumerate(epochs):
    r_epoch = resp_dict[epoch]
    time_bins = r_epoch.shape[-1]
    if i == 0:
        X = r_epoch
    else:
        X = np.append(X, r_epoch, axis=-1)

X_trial_average = X.mean(axis=0)
nreps = X.shape[0]
nstim = X.shape[-1]
n_cells = X.shape[1]

# build Y matrix of one hot vectors
Y = np.zeros((nstim, nstim * nreps))
for stim in range(nstim):
    yt = np.zeros((nreps, nstim))
    yt[:, stim] = 1
    yt = yt.reshape(1, -1)
    Y[stim, :] = yt

# unravel X matrix to be neurons X reps * stimuli
X = X.transpose(1, 0, -1).reshape(X.shape[1], -1)

# preprocess X data (z score)
X_scale = scale(X, axis=-1)

# ================== perform dimensionality reduction with PLS regression ====================
# ======================== this works very poorly for many stimuli ===========================
# tricky thing is, it maxmizes covariance with a reduced representation of Y too, which makes
# it a little tricky to interpret, but I guess looking at the y_weights tells you which stimuli
# drives the most variance in X
pls = PLSRegression(n_components=2, max_iter=100, tol=1e-7, scale=False)
pls.fit(X_scale.T, Y.T)

# reshape projections to ID stimuli, then plot the data in reduced space
X_new = pls.x_scores_.reshape(2, nreps, nstim)
Y_new = pls.y_scores_.reshape(2, nreps, nstim)

f, ax = plt.subplots(1, 2, figsize=(10, 5))
for s in range(nstim):

    ax[0].plot(X_new[0, :, s], X_new[1, :, s], '.')
    el = compute_ellipse(X_new[0, :, s], X_new[1, :, s])
    ax[0].plot(el[0], el[1], color=ax[0].get_lines()[-1].get_color())

    ax[1].scatter(Y_new[0, :, s], Y_new[1, :, s], s=10)

f.suptitle('PLS on single trial data')
ax[0].set_xlabel('PLS 1')
ax[0].set_ylabel('PLS 2')
ax[0].set_title('X data')
ax[1].set_title('Y data')

# ============== perform dimensionality reduction with trial averaged PCA ====================
X_center = X_trial_average - X_trial_average.mean(axis=-1, keepdims=True)
pca = PCA(n_components=2)
pca.fit(X_center.T)

# project single trial data
X_pca = X.T @ pca.components_.T
X_pca = X_pca.T.reshape(2, nreps, nstim)

f, ax = plt.subplots(1, 1, figsize=(5, 5))
for s in range(nstim):
    ax.plot(X_pca[0, :, s], X_pca[1, :, s], '.')
    el = compute_ellipse(X_pca[0, :, s], X_pca[1, :, s])
    ax.plot(el[0], el[1], color=ax.get_lines()[-1].get_color())

ax.set_title('Trial-averaged PCA')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.show()