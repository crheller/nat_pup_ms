"""
First step, do PCA on residual to get components that reflect correlated activity.

Next, do 1st order PLS to find pupil dims.

Finally, tranform PCs to power and do PLS again to determine which dims 
have power that correlates with pupil.

Idea behind this is to avoid the complications of trying to do some sort of PLS
with the power of the spikes.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from scipy.optimize import nnls
import scipy.ndimage.filters as sf

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp, sliding_window
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.preprocessing as cpreproc

from nems_lbhb.baphy import parse_cellid

site = 'TAR010c' #'DRX006b.e65:128'
batch = 289
fs = 4
ops = {'batch': batch, 'cellid': site}
xmodel = 'ns.fs{}.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'.format(fs)
path = '/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/'
low = 0.5
high = 2

cells, _ = parse_cellid(ops)
rec = generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                    cache_path=path, recache=False)

epochs = [e for e in rec.epochs.name.unique() if 'STIM' in e]
dresp = rec['resp'].extract_epochs(epochs)
zresp = cpreproc.zscore_per_stim(dresp, d2=None)
zresp = rec['resp'].replace_epochs(zresp, mask=rec['mask'])
rec['zresp'] = zresp

ff_residuals = rec['zresp']._data.copy()
nan_idx = np.isnan(ff_residuals[0, :])
ff_residuals[:, nan_idx] = 0
ff_residuals = bandpass_filter_resp(ff_residuals, low, high, fs=fs, boxcar=True)
rec['ff_residuals'] = rec['resp']._modified_copy(ff_residuals)
rec = rec.apply_mask()

raw_residual = rec['zresp']._data
pupil = rec['pupil']._data

# first, do full PCA on residuals
#raw_residual = scale(raw_residual, with_mean=True, with_std=True, axis=-1)
pca = PCA()
pca.fit(raw_residual.T)
pca_transform = raw_residual.T.dot(pca.components_.T).T

# do first order regression
X = pca_transform
X = scale(X, with_mean=True, with_std=True, axis=-1)
y = scale(pupil, with_mean=True, with_std=True, axis=-1)
lr = LinearRegression(fit_intercept=True)
lr.fit(X.T, y.squeeze())
first_order_weights = lr.coef_
fow_norm = lr.coef_ / np.linalg.norm(lr.coef_)

# do second order regression
ff = rec['ff_residuals']._data
# get power of each neuron
power = []
for n in range(ff.shape[0]):
    _ff = ff[[n], :]
    t, _ffsw = sliding_window(_ff, fs=fs, window_size=4, step_size=2)
    _ffpower = np.sum(_ffsw**2, axis=-1) / _ffsw.shape[-1]
    power.append(_ffpower)
power = np.stack(power)
t, _p = sliding_window(pupil, fs=fs, window_size=4, step_size=2)
pds = np.mean(_p, axis=-1)[np.newaxis, :]

power = scale(power, with_mean=True, with_std=True, axis=-1)
pds = scale(pds, with_mean=True, with_std=True, axis=-1)

# do nnls regression to avoid to sign ambiguity due to power conversion
x, r = nnls(-power.T, pds.squeeze())
second_order_weights = x

if np.linalg.norm(x)==0:
    sow_norm = x
else:
    sow_norm = x / np.linalg.norm(x)

# project weights back into neuron space (then the can be compared with PC weights too)
fow_nspace = pca.components_.T.dot(fow_norm)
sow_nspace = pca.components_.T.dot(sow_norm)

# now, compare the two dimensions, 
# the amount of variance explained by each
# and the overall PCA spcae

# first plot timeseries onto the two dims
sigma = 1
f, ax = plt.subplots(2, 1, figsize=(8, 4))

ax[0].plot(pupil.T, color='purple')

#ax[2].plot(sf.gaussian_filter1d(X.T.dot(sow_norm), sigma), color='purple')
ax[1].scatter(range(ff.shape[-1]), ff.T.dot(sow_norm), c=pupil.squeeze(), cmap='Purples', vmin=pupil.min()-2, s=5)
#ax[1].scatter(range(ff.shape[-1]), X.T.dot(sow_nspace), c=pupil.squeeze(), cmap='Purples', vmin=pupil.min()-2, s=5)

f.tight_layout()

# compare first / second order weights
f, ax = plt.subplots(2, 1, figsize=(6, 6))

ax[0].scatter(fow_nspace, sow_nspace, edgecolor='white')
ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0, linestyle='--', color='k')

ax[0].set_xlabel('First-order weights')
ax[0].set_ylabel('Second-order weights')

ax[0].set_title(r"$\mathbf{x} \cdot \mathbf{y} = %s $" % round(fow_norm.dot(sow_norm), 2))

ax[0].axis('square')

# scree plot of PCA variance, as well as fraction variance along each
# of the pupil dimensions
var_1st_order = np.var(pca_transform.T.dot(fow_norm)[:, np.newaxis] @ fow_norm[np.newaxis,:]) / np.var(pca_transform)
#var_1st_order = (np.var(pca_transform) - np.var(cor_residual)) / np.var(pca_transform) 
var_2nd_order = np.var(pca_transform.T.dot(sow_norm)[:, np.newaxis] @ sow_norm[np.newaxis,:]) / np.var(pca_transform)
var_explained = np.append(pca.explained_variance_ratio_, [var_1st_order, var_2nd_order])
idx = np.argsort(var_explained)[::-1]
fo_idx = np.argwhere(var_explained[idx]==var_1st_order)[0][0]
so_idx = np.argwhere(var_explained[idx]==var_2nd_order)[0][0]

ax[1].bar(range(len(var_explained)), var_explained[idx], edgecolor='k', width=0.7, color='lightgrey', label='PCs')
ax[1].bar(fo_idx, var_explained[idx][fo_idx], color='forestgreen', edgecolor='k', width=0.7, label='First-order pupil')
ax[1].bar(so_idx, var_explained[idx][so_idx], color='purple', edgecolor='k', width=0.7, label='Second-order pupil')
ax[1].set_ylabel('Fraction variance explained')
ax[1].set_xlabel('PC')

ax[1].legend(frameon=False)

f.tight_layout()

plt.show()


