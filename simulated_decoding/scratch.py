import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.plotting as cplt

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'svg.fonttype': 'none'})


np.random.seed(123)

Ndim = 25
Ntrials= 100
var_ratio = 3 # pc1 has X times the variance as pc2

# simulated data
u1 = 4
u2 = 8
u = np.stack((np.random.poisson(u1, Ndim), np.random.poisson(u2, Ndim)))

# make two dimensional noise:
# one large dim ~orthogonal to dU and one smaller dim ~ parallel to dU
dU = u[[1], :] - u[[0], :]
dU = dU / np.linalg.norm(dU)

diff_cor = dU + np.random.normal(0, 0.001, dU.shape)
diff_cor = diff_cor / np.linalg.norm(diff_cor) * 5 
pc1 = np.random.normal(0, 1, dU.shape)
pc1 = (pc1 / np.linalg.norm(pc1)) * 5 * var_ratio

evecs = np.concatenate((diff_cor, pc1), axis=0)
cov = evecs.T.dot(evecs)

# simulate full data matrix
_X = np.random.multivariate_normal(np.zeros(Ndim), cov, Ntrials)
X1 = _X + u[0, :]
X2 = _X + u[1, :]
X_raw = np.stack((X1, X2)).transpose([-1, 1, 0])

# add random noise to data matrix
X_raw += np.random.normal(0, 0.5, X_raw.shape)

# preprocess data (zscore)
X, _ = nat_preproc.scale_est_val([X_raw], [X_raw])
X = X[0]

# plot variance explained over PCs in the full datset to illustrate noise structure
pca = PCA()
pca.fit(X[:, :, 0].T)

f, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()

ax[0].plot(np.cumsum(pca.explained_variance_ratio_), 'o-', color='k')
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].set_ylabel('Explained variance')
ax[0].set_xlabel('PC')

# reduce dimensionality and plot in TDR dimensions
Y = dr.get_one_hot_matrix(2, Ntrials)
Xflat = nat_preproc.flatten_X(X[:, :, :, np.newaxis])
tdr = dr.TDR()
tdr.fit(Xflat.T, Y.T)
Xtdr = Xflat.T.dot(tdr.weights.T).T
Xtdr = nat_preproc.fold_X(Xtdr, nreps=Ntrials, nstim=2, nbins=1).squeeze()

# run dprime analysis on the reduced data, and on the raw data
dp_tdr, wopt_tdr, evals_tdr, evecs_tdr, dU_tdr = decoding.compute_dprime(Xtdr[:, :, 0], Xtdr[:, :, 1])
dp, wopt, evals, evecs, dU = decoding.compute_dprime(X[:, :, 0], X[:, :, 1])
el1 = cplt.compute_ellipse(Xtdr[0, :, 0], Xtdr[1, :, 0])
el2 = cplt.compute_ellipse(Xtdr[0, :, 1], Xtdr[1, :, 1])

ax[1].scatter(Xtdr[0, :, 0], Xtdr[1, :, 0], s=20, color='r', edgecolor='white', alpha=0.5)
ax[1].scatter(Xtdr[0, :, 1], Xtdr[1, :, 1], s=20, color='b', edgecolor='white', alpha=0.5)
ax[1].plot(el1[0], el1[1], color='r', lw=2)
ax[1].plot(el2[0], el2[1], color='b', lw=2)

# plot wopt 
wopt_unit = wopt_tdr / np.linalg.norm(wopt_tdr) * (np.linalg.norm(dU_tdr) / 2)
wopt_raw = (wopt / np.linalg.norm(wopt)).T.dot(tdr.weights.T).T
wopt_raw = wopt_raw / np.linalg.norm(wopt_raw) * (np.linalg.norm(dU_tdr) / 2)
ax[1].plot([0, wopt_unit[0]], [0, wopt_unit[1]], 'k-', lw=2, label=r"$w_{opt, tdr}$")
ax[1].plot([0, -wopt_unit[0]], [0, -wopt_unit[1]], 'k-', lw=2)
ax[1].plot([0, wopt_raw[0]], [0, wopt_raw[1]], color='orange', lw=2, label=r"$w_{opt, raw}$")
ax[1].plot([0, -wopt_raw[0]], [0, -wopt_raw[1]], color='orange', lw=2)

# plot first noise pc (for reduced space, and the projection from the high d space)
noise_tdr = evecs_tdr[:, 0] / np.linalg.norm(evecs_tdr[:, 0]) * (np.linalg.norm(dU_tdr) / 2)
noise_raw = evecs[:, 0].dot(tdr.weights.T)
noise_raw = noise_raw / np.linalg.norm(noise_raw) * (np.linalg.norm(dU_tdr) / 2)
ax[1].plot([0, noise_tdr[0]], [0, noise_tdr[1]], color='grey', lw=2, label=r"$\mathbf{e}_{1, tdr}$")
ax[1].plot([0, -noise_tdr[0]], [0, -noise_tdr[1]], color='grey', lw=2)
ax[1].plot([0, noise_raw[0]], [0, noise_raw[1]], linestyle='dotted', color='purple', lw=2, label=r"$\mathbf{e}_{1, raw}$")
ax[1].plot([0, -noise_raw[0]], [0, -noise_raw[1]], linestyle='dotted', color='purple', lw=2)

ax[1].set_xlabel(r"$\Delta \mathbf{\mu}$")
ax[1].set_ylabel("TDR 2")
ax[1].set_title(r"$d'^{2}_{tdr} = %s, \Delta \mathbf{\mu} = %s$" 
                "\n"
                r"$d'^{2}_{raw} = %s, \Delta \mathbf{\mu} = %s$" % (round(dp_tdr), round(np.linalg.norm(dU_tdr), 2), 
                                          round(dp), round(np.linalg.norm(dU), 2)))
ax[1].legend()
ax[1].axis('square')

# plot this on its own too, just for the sake of visualization
_f, _ax = plt.subplots(1, 1, figsize=(4, 4))
_ax.scatter(Xtdr[0, :, 0], Xtdr[1, :, 0], s=20, color='r', edgecolor='white', alpha=0.5)
_ax.scatter(Xtdr[0, :, 1], Xtdr[1, :, 1], s=20, color='b', edgecolor='white', alpha=0.5)
_ax.plot(el1[0], el1[1], color='r', lw=2)
_ax.plot(el2[0], el2[1], color='b', lw=2)

# plot wopt 
'''
wopt_unit = wopt_tdr / np.linalg.norm(wopt_tdr) * (np.linalg.norm(dU_tdr) / 2)
wopt_raw = (wopt / np.linalg.norm(wopt)).T.dot(tdr.weights.T).T
wopt_raw = wopt_raw / np.linalg.norm(wopt_raw) * (np.linalg.norm(dU_tdr) / 2)
_ax.plot([0, wopt_unit[0]], [0, wopt_unit[1]], 'k-', lw=2, label=r"$w_{opt, tdr}$")
_ax.plot([0, -wopt_unit[0]], [0, -wopt_unit[1]], 'k-', lw=2)
_ax.plot([0, wopt_raw[0]], [0, wopt_raw[1]], color='orange', lw=2, label=r"$w_{opt, raw}$")
_ax.plot([0, -wopt_raw[0]], [0, -wopt_raw[1]], color='orange', lw=2)
'''

# plot first noise pc (for reduced space, and the projection from the high d space)
noise_tdr = evecs_tdr[:, 0] / np.linalg.norm(evecs_tdr[:, 0]) * (np.linalg.norm(dU_tdr) / 2) * 3
noise_raw = evecs[:, 0].dot(tdr.weights.T)
noise_raw = noise_raw / np.linalg.norm(noise_raw) * (np.linalg.norm(dU_tdr) / 2) * 3
_ax.plot([0, noise_tdr[0]], [0, noise_tdr[1]], color='grey', lw=5, label=r"$\mathbf{e}_{1, tdr}$")
_ax.plot([0, -noise_tdr[0]], [0, -noise_tdr[1]], color='grey', lw=5)
_ax.plot([0, noise_raw[0]], [0, noise_raw[1]], linestyle='dotted', color='magenta', lw=2, label=r"$\mathbf{e}_{1, raw}$")
_ax.plot([0, -noise_raw[0]], [0, -noise_raw[1]], linestyle='dotted', color='magenta', lw=2)

_ax.set_xlabel(r"$\Delta \mathbf{\mu}$")
_ax.set_ylabel("TDR 2")
_ax.set_title(r"$d'^{2}_{tdr} = %s, \Delta \mathbf{\mu} = %s$" 
                "\n"
                r"$d'^{2}_{raw} = %s, \Delta \mathbf{\mu} = %s$" % (round(dp_tdr), round(np.linalg.norm(dU_tdr), 2), 
                                          round(dp), round(np.linalg.norm(dU), 2)))
_ax.legend(frameon=False)
_ax.axis('square')
 
_f.tight_layout()

# ====================== perform same analysis as above, but use cross-validation ======================
# determine if decoding in full space is overfitting
njacks = 5
X = X_raw

# generate est / val
est, val = nat_preproc.get_est_val_sets(X, njacks=njacks)
est, val = nat_preproc.scale_est_val(est, val)

tdr_test = np.zeros(njacks)
tdr_train = np.zeros(njacks)
test = np.zeros(njacks)
train = np.zeros(njacks)
for j in range(njacks):
    y = dr.get_one_hot_matrix(2, int(Ntrials / 2))
    xtrain = nat_preproc.flatten_X(est[j][:, :, :, np.newaxis])
    xtest = nat_preproc.flatten_X(val[j][:, :, :, np.newaxis])

    tdr = dr.TDR()
    tdr.fit(xtrain.T, y.T)
    Xtdr_train = xtrain.T.dot(tdr.weights.T).T
    Xtdr_train = nat_preproc.fold_X(Xtdr_train, nreps=int(Ntrials/2), nstim=2, nbins=1).squeeze()

    Xtdr_test = xtest.T.dot(tdr.weights.T).T
    Xtdr_test = nat_preproc.fold_X(Xtdr_test, nreps=int(Ntrials/2), nstim=2, nbins=1).squeeze()

    # run dprime analysis on the reduced data
    dp_tdr_train, wopt_tdr_train, evals_tdr_train, evecs_tdr_train, dU_tdr_train = \
                        decoding.compute_dprime(Xtdr_train[:, :, 0], Xtdr_train[:, :, 1])
    dp_tdr_test, wopt_tdr_test, evals_tdr_test, evecs_tdr_test, dU_tdr_test = \
                        decoding.compute_dprime(Xtdr_test[:, :, 0], Xtdr_test[:, :, 1], wopt=wopt_tdr_train)
    tdr_test[j] = dp_tdr_test
    tdr_train[j] = dp_tdr_train

    # run dprime on raw data
    dp_train, wopt_train, evals_train, evecs_train, dU_train = decoding.compute_dprime(est[j][:, :, 0], est[j][:, :, 1])
    dp_test, wopt_test, evals_test, evecs_test, dU_test = decoding.compute_dprime(val[j][:, :, 0], val[j][:, :, 1], wopt=wopt_train)

    test[j] = dp_test
    train[j] = dp_train

ax[2].errorbar([0, 1], [train.mean(), test.mean()], 
                    yerr=[train.std() / np.sqrt(njacks), test.std() / np.sqrt(njacks)], lw=2,
                    linestyle='none', marker='o', markersize=10, color='k')
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(['Train', 'Test'])
ax[2].set_ylabel(r"$d'^{2}$")
ax[2].set_title('Raw')

ax[3].errorbar([0, 1], [tdr_train.mean(), tdr_test.mean()], 
                    yerr=[tdr_train.std() / np.sqrt(njacks), tdr_test.std() / np.sqrt(njacks)], lw=2,
                    linestyle='none', marker='o', markersize=10, color='k')
ax[3].set_xticks([0, 1])
ax[3].set_xticklabels(['Train', 'Test'])
ax[3].set_ylabel(r"$d'^{2}$")
ax[3].set_title('TDR')

m = np.max(ax[2].get_ylim() + ax[3].get_ylim())
mi = np.min(ax[2].get_ylim() + ax[3].get_ylim())

ax[3].set_ylim((mi, m))
ax[2].set_ylim((mi, m))


f.tight_layout()

plt.show()