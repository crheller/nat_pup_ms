"""
Use big - small noise correlations to define LV axis.
Subtract it out, recompute noise correlations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp, sliding_window
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.preprocessing as cpreproc

from nems_lbhb.baphy import parse_cellid
from nems_lbhb.preprocessing import create_pupil_mask

np.random.seed(123)

site = 'BOL006b' #'BOL006b' #'DRX006b.e65:128'
batch = 294

fs = 4
ops = {'batch': batch, 'cellid': site}
xmodel = 'ns.fs{}.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'.format(fs)
path = '/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/'
low = 0.5
high = 2  # for filtering the projection

cells, _ = parse_cellid(ops)
rec = generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                    cache_path=path, recache=False)
#fill = rec['resp']._data.copy()
#idx = np.argwhere(np.isnan(fill[0,:]))
#fill[:, idx] = 0
#rec['resp'] = rec['resp']._modified_copy(fill)
#rec = cpreproc.bandpass_filter_resp(rec, low_c=low, high_c=high)
rec = rec.apply_mask(reset_epochs=True)
pupil = rec['pupil']._data.squeeze()
epochs = [e for e in rec.epochs.name.unique() if 'STIM' in e]
rec['resp2'] = rec['resp']._modified_copy(rec['resp']._data)
rec['pupil2'] = rec['pupil']._modified_copy(rec['pupil']._data)

# ===================================== perform analysis on raw data =======================================
rec_bp = rec.copy()
ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
rec_bp = create_pupil_mask(rec_bp, **ops)
ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
rec_sp = rec.copy()
rec_sp = create_pupil_mask(rec_sp, **ops)

real_dict_small = rec_sp['resp'].extract_epochs(epochs, mask=rec_sp['mask'])
real_dict_big = rec_bp['resp'].extract_epochs(epochs, mask=rec_bp['mask'])
real_dict_all = rec['resp'].extract_epochs(epochs)
pred_dict_all = rec['psth'].extract_epochs(epochs)

real_dict_small = cpreproc.zscore_per_stim(real_dict_small, d2=real_dict_small, with_std=False)
real_dict_big = cpreproc.zscore_per_stim(real_dict_big, d2=real_dict_big, with_std=False)
real_dict_all = cpreproc.zscore_per_stim(real_dict_all, d2=real_dict_all, with_std=False)
pred_dict_all = cpreproc.zscore_per_stim(pred_dict_all, d2=pred_dict_all, with_std=False)

eps = list(real_dict_big.keys())
nCells = real_dict_big[eps[0]].shape[1]
for i, k in enumerate(real_dict_big.keys()):
    if i == 0:
        resp_matrix = np.transpose(real_dict_all[k], [1, 0, -1]).reshape(nCells, -1)
        resp_matrix_small = np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)
        resp_matrix_big = np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)
        pred_matrix = np.transpose(pred_dict_all[k], [1, 0, -1]).reshape(nCells, -1)
    else:
        resp_matrix = np.concatenate((resp_matrix, np.transpose(real_dict_all[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
        resp_matrix_small = np.concatenate((resp_matrix_small, np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
        resp_matrix_big = np.concatenate((resp_matrix_big, np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
        pred_matrix = np.concatenate((pred_matrix, np.transpose(pred_dict_all[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)

nc_resp_small = resp_matrix_small #cpreproc.bandpass_filter_resp(resp_matrix_small, low_c=low, high_c=high, fs=fs)
nc_resp_big = resp_matrix_big #cpreproc.bandpass_filter_resp(resp_matrix_big, low_c=low, high_c=high, fs=fs)
small = np.cov(nc_resp_small)
np.fill_diagonal(small, 0)
big = np.cov(nc_resp_big)
np.fill_diagonal(big, 0)
diff = small - big
evals, evecs = np.linalg.eig(diff)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

# project rank1 data onto first eval of diff
r1_resp = (resp_matrix.T.dot(evecs[:, 0])[:, np.newaxis] @ evecs[:, [0]].T).T

# compute PCs over all the data
pca = PCA()
pca.fit(resp_matrix.T)

# compute variance of rank1 matrix along each PC
var = np.zeros(resp_matrix.shape[0])
fo_var = np.zeros(pred_matrix.shape[0])
for pc in range(0, resp_matrix.shape[0]):
    var[pc] = np.var(r1_resp.T.dot(pca.components_[pc])) / np.sum(pca.explained_variance_)
    fo_var[pc] = np.var(pred_matrix.T.dot(pca.components_[pc])) / np.sum(pca.explained_variance_)

# project onto first evec and filter
#rm = cpreproc.bandpass_filter_resp(resp_matrix, low_c=low, high_c=high, fs=fs)
proj = resp_matrix.T.dot(evecs[:, 0])[np.newaxis]
proj_filt = proj.copy()
proj_filt = cpreproc.bandpass_filter_resp(proj, low_c=low, high_c=high, fs=fs)

# caculate power in the filtered residual
t, ffsw = sliding_window(proj_filt, fs=fs, window_size=4, step_size=1) 
power = np.sum(ffsw**2, axis=-1) / ffsw.shape[-1] 

t, pds = sliding_window(pupil[np.newaxis, :], fs=fs, window_size=4, step_size=1)
pds = pds.mean(axis=-1)

# plot results
ncells = resp_matrix.shape[0]
f = plt.figure(figsize=(15, 6))
espec = plt.subplot2grid((2, 5), (0, 0))
pcvar = plt.subplot2grid((2, 5), (0, 3), colspan=2)
pax = plt.subplot2grid((2, 5), (1, 0), colspan=3)
betaax = plt.subplot2grid((2, 5), (0, 1))
pcvar1 = plt.subplot2grid((2, 5), (1, 3), colspan=2)

x = np.arange(0, len(evals)) - int(len(evals)/2)
espec.plot(x, evals, '.-', label='raw data')
espec.axhline(0, linestyle='--', color='grey')
espec.axvline(0, linestyle='--', color='grey')
espec.set_title("Eigenspectrum of "
                    "\n"
                    r"$\Sigma_{small} - \Sigma_{large}$")
espec.set_ylabel(r"$\lambda$")
espec.set_xlabel(r"$\alpha$")

# project onto PCs and make the PCA variance plot
pcvar.bar(range(0, ncells), pca.explained_variance_ratio_, edgecolor='k', width=0.7, color='lightgrey', label='Raw data')
pcvar.bar(range(0, ncells),var, edgecolor='k', width=0.7, color='tab:blue', label=r"$\mathbf{r}^{T}\mathbf{e}_{\alpha=1}$")
pcvar.set_ylabel('Explained \n variance')
pcvar.set_xlabel('PC')
pcvar.legend(frameon=False)
pcvar.set_title(r"$\beta_{2}$")

t1 = np.linspace(0, 100, proj_filt.shape[-1])
pax.scatter(t1, proj_filt.squeeze(), c=pupil+2, vmin=pupil.min(), cmap='Purples', s=15)
pax.set_ylabel(r"$\mathbf{r}^{T}\mathbf{e}_{\alpha=1}$")
pax.set_title("Projection onto first dimension of difference covariance matrix")
pax.set_xlabel('Time')


# ============================== shuffle pupil and repeat ====================================
# in order to show the "null" distribution of eigenvalues
pupil = rec['pupil']._data.copy().squeeze()
np.random.shuffle(pupil)
rec['pupil'] = rec['pupil']._modified_copy(pupil[np.newaxis, :])

rec_bp = rec.copy()
ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
rec_bp = create_pupil_mask(rec_bp, **ops)
ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
rec_sp = rec.copy()
rec_sp = create_pupil_mask(rec_sp, **ops)

shuf_dict_small = rec_sp['resp'].extract_epochs(epochs, mask=rec_sp['mask'])
shuf_dict_big = rec_bp['resp'].extract_epochs(epochs, mask=rec_bp['mask'])

shuf_dict_small = cpreproc.zscore_per_stim(shuf_dict_small, d2=shuf_dict_small, with_std=True)
shuf_dict_big = cpreproc.zscore_per_stim(shuf_dict_big, d2=shuf_dict_big, with_std=True)

eps = list(shuf_dict_big.keys())
nCells = shuf_dict_big[eps[0]].shape[1]
for i, k in enumerate(shuf_dict_big.keys()):
    if i == 0:
        shuf_matrix_small = np.transpose(shuf_dict_small[k], [1, 0, -1]).reshape(nCells, -1)
        shuf_matrix_big = np.transpose(shuf_dict_big[k], [1, 0, -1]).reshape(nCells, -1)
    else:
        shuf_matrix_small = np.concatenate((shuf_matrix_small, np.transpose(shuf_dict_small[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
        shuf_matrix_big = np.concatenate((shuf_matrix_big, np.transpose(shuf_dict_big[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)

shuf_small = np.corrcoef(shuf_matrix_small)
shuf_big = np.corrcoef(shuf_matrix_big)
shuf_diff = shuf_small - shuf_big
shuf_evals, shuf_evecs = np.linalg.eig(shuf_diff)
shuf_evals = shuf_evals[np.argsort(shuf_evals)[::-1]]

espec.plot(x, shuf_evals, '.-', label='shuffle pupil')
espec.legend(frameon=False)


# =================================== Find first order dimension =================================
# do this by getting mean zscore rate in large minus small, norm this difference vector. 
residual = rec['resp2']._data - rec['psth_sp']._data  # get rid of stimulus information
# zscore residual
residual = residual - residual.mean(axis=-1, keepdims=True)
#residual = residual / residual.std(axis=-1, keepdims=True)
rec['residual'] = rec['resp']._modified_copy(residual)
rec['pupil'] = rec['pupil']._modified_copy(rec['pupil2']._data)

# get large and small pupil means
rec = rec.create_mask(True)
ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
rec = create_pupil_mask(rec, **ops)

large = rec.apply_mask()['residual']._data

rec = rec.create_mask(True)
ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
rec = create_pupil_mask(rec, **ops)
small = rec.apply_mask()['residual']._data

beta1 = large.mean(axis=-1) - small.mean(axis=-1)
beta1 = beta1 / np.linalg.norm(beta1)

# project rank1 data onto beta1
r1_resp = (resp_matrix.T.dot(beta1)[:, np.newaxis] @ beta1[:, np.newaxis].T).T

# get fo variance along each PC
var1 = np.zeros(resp_matrix.shape[0])
for pc in range(0, resp_matrix.shape[0]):
    var1[pc] = np.var(r1_resp.T.dot(pca.components_[pc])) / np.sum(pca.explained_variance_)

# project onto PCs and make the PCA variance plot
pcvar1.bar(range(0, ncells), pca.explained_variance_ratio_, edgecolor='k', width=0.7, color='lightgrey', label='Raw data')
pcvar1.bar(range(0, ncells),var1, edgecolor='k', width=0.7, color='tab:orange', label=r"$\mathbf{r}^{T}\beta_{1}$")
pcvar1.set_ylabel('Explained \n variance')
pcvar1.set_xlabel('PC')
pcvar1.legend(frameon=False)
pcvar1.set_title(r"$\beta_{1}$")

# plot first order vs. second order weights
betaax.scatter(beta1, evecs[:, 0], color='grey', edgecolor='white', s=25)
betaax.axhline(0, linestyle='--', color='k')
betaax.axvline(0, linestyle='--', color='k')
betaax.set_xlabel(r"$\beta_{1}$")
betaax.set_ylabel(r"$\beta_{2}$")
betaax.set_title(r"$r=%s$" % (round(np.corrcoef(beta1, evecs[:, 0])[0, 1], 2)))


f.tight_layout()

plt.show()
