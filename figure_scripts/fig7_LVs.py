"""
Illustrate 1st / 2nd order pupil axes over all the data, and their relation with 
the first noise PC, as a point of reference.
"""
import colors as color

import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'svg.fonttype': 'none'})

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp, sliding_window
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc

from nems_lbhb.baphy import parse_cellid

savefig = True
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig7_LVs.svg'
# set up figure
f = plt.figure(figsize=(12, 7.2))

pupax = plt.subplot2grid((3, 5), (0, 0), colspan=3)
lvax = plt.subplot2grid((3, 5), (1, 0), colspan=3)
betaax = plt.subplot2grid((3, 5), (0, 3))
pcaax = plt.subplot2grid((3, 5), (1, 3), colspan=2)
varax = plt.subplot2grid((3, 5), (2, 3))
cosax = plt.subplot2grid((3, 5), (2, 4))
cartax = plt.subplot2grid((3, 5), (2, 2))

# =========================== LOAD / PLOT RESULTS OVER ALL SITES =========================
fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/pca_regression_lvs.pickle'
good_sites = True
high_rep_sites = ['TAR010c', 'TAR017b', 
                'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
                'DRX007a.e1:64', 'DRX007a.e65:128', 
                'DRX008b.e1:64', 'DRX008b.e65:128',
                'BOL005c', 'BOL006b']

# load results from pickle file
with open(fn, 'rb') as handle:
    results = pickle.load(handle)

# LOAD VARIANCE EXPLAINED

# get all PC1 var explained
pc1_var = [results[s]['pc_variance'][0] for s in results.keys()]
# get all 1st order pupil var explained
fo_variance = [results[s]['var_1st_order'] for s in results.keys()]
# get all 2nd order pupil var explained
so_variance = [results[s]['var_2nd_order'] for s in results.keys()]

# LOAD COSINE SIMILARITY

# pc1 vs. first order weights
pc1_fow = [abs(results[s]['cos_fow_PC1']) for s in results.keys()]
# pc1 vs. second order weights
pc1_sow = [abs(results[s]['cos_sow_PC1']) for s in results.keys()]
# first order weights vs. second order weights
fow_sow = [abs(results[s]['cos_fow_sow']) for s in results.keys()]

n_neurons = [results[s]['fow'].shape[0] for s in results.keys()]

# pack all results into df for easy plotting
data = np.stack([pc1_var, fo_variance, so_variance, pc1_fow, pc1_sow, fow_sow, n_neurons]).T
df = pd.DataFrame(columns=['pc1_var', 'fo_var', 'so_var', 'pc1_fow', 'pc1_sow', 'fow_sow', 'n_neurons'],
                  index=list(results.keys()), data=data)

if good_sites:
    df = df.loc[high_rep_sites]


df[['pc1_var', 'fo_var', 'so_var']].T.plot(ax=varax, legend=False, color='lightgrey')
df[['pc1_var', 'fo_var', 'so_var']].mean().plot(ax=varax, color='k', lw=2, marker='o')
varax.axhline(0, linestyle='--', color='grey')
varax.set_ylabel('Fraction \n explained variance')
varax.set_xticks([0, 1, 2])
varax.set_xticklabels([r'$\mathbf{PC}_{1}$', r'$\mathbf{\beta}_{1}$', r'$\mathbf{\beta}_{2}$'])

df[['pc1_fow', 'pc1_sow', 'fow_sow']].T.plot(ax=cosax, legend=False, alpha=0.7, color='lightgrey')
df[['pc1_fow', 'pc1_sow', 'fow_sow']].mean().plot(ax=cosax, color='k', lw=2, marker='o')
cosax.axhline(0, linestyle='--', color='grey')
cosax.set_ylabel(r"$cos(\theta)$")
cosax.set_xticks([0, 1, 2])
cosax.set_xticklabels([r'$\mathbf{PC}_{1} \cdot \mathbf{\beta}_{1}$', 
                        r'$\mathbf{PC}_{1} \cdot \mathbf{\beta}_{2}$', 
                        r'$\mathbf{\beta}_{1} \cdot \mathbf{\beta}_{2}$'])

# ============================= LOAD / ANALYZE / PLOT EXAMPLE SITE ==============================
site = 'DRX006b.e65:128' #'DRX006b.e65:128'
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
ff_residuals = rec['resp']._data - rec['psth_sp']._data
nan_idx = np.isnan(ff_residuals[0, :])
ff_residuals[:, nan_idx] = 0
ff_residuals = bandpass_filter_resp(ff_residuals, low, high, fs=fs, boxcar=True)
rec['ff_residuals'] = rec['resp']._modified_copy(ff_residuals)
rec = rec.apply_mask()

pupil = rec['pupil']._data
raw_residual = rec['resp']._data - rec['psth_sp']._data
cor_residual = rec['resp']._data - rec['psth']._data

# first, do full PCA on residuals
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
time = np.linspace(0, pupil.shape[-1] / fs, pupil.shape[-1])
pupax.plot(time, pupil.T, color='purple')
pupax.set_ylabel("Pupil size")

#ax[2].plot(sf.gaussian_filter1d(X.T.dot(sow_norm), sigma), color='purple')
lvax.scatter(time, ff.T.dot(sow_norm), c=pupil.squeeze(), cmap='Purples', vmin=pupil.min()-2, s=5)
lvax.set_xlabel('Time (s)')
lvax.set_ylabel(r'$\mathbf{r}(t)^{T} \mathbf{\beta}_{2}$')
lvax.set_title(r'Projection of residuals onto $\mathbf{\beta}_{2}$')

# compare first / second order weights
betaax.scatter(fow_nspace, sow_nspace, edgecolor='white', color='grey')
betaax.axhline(0, linestyle='--', color='k')
betaax.axvline(0, linestyle='--', color='k')

betaax.set_xlabel(r'$\mathbf{\beta}_{1}$')
betaax.set_ylabel(r'$\mathbf{\beta}_{2}$')

betaax.set_title(r"$cos(\theta_{\mathbf{\beta}_{1}, \mathbf{\beta}_{2}}) = %s $" % round(fow_norm.dot(sow_norm), 2))

betaax.axis('square')

# scree plot of PCA variance, as well as fraction variance along each
# of the pupil dimensions
var_1st_order = np.var(pca_transform.T.dot(fow_norm)[:, np.newaxis] @ fow_norm[np.newaxis,:]) / np.var(pca_transform)
#var_1st_order = (np.var(pca_transform) - np.var(cor_residual)) / np.var(pca_transform) 
var_2nd_order = np.var(pca_transform.T.dot(sow_norm)[:, np.newaxis] @ sow_norm[np.newaxis,:]) / np.var(pca_transform)
var_explained = np.append(pca.explained_variance_ratio_, [var_1st_order, var_2nd_order])
idx = np.argsort(var_explained)[::-1]
fo_idx = np.argwhere(var_explained[idx]==var_1st_order)[0][0]
so_idx = np.argwhere(var_explained[idx]==var_2nd_order)[0][0]

pcaax.bar(range(len(var_explained)), var_explained[idx], edgecolor='k', width=0.7, color='lightgrey', label='PCs')
pcaax.bar(fo_idx, var_explained[idx][fo_idx], color=color.RAW, edgecolor='k', width=0.7, label=r'$\mathbf{\beta}_{1}$')
pcaax.bar(so_idx, var_explained[idx][so_idx], color=color.CORRECTED, edgecolor='k', width=0.7, label=r'$\mathbf{\beta}_{2}$')
pcaax.set_ylabel('Fraction \n explained variance')
pcaax.set_xlabel('PC')

pcaax.legend(frameon=False)


# simulated illustration of 1st order, 2nd order, and PC1 axes 
# in 2D space
cov = np.array([[1, 0.6], [0.6, 1]])
data = np.random.multivariate_normal([0, 0], cov, (100,))
pca = PCA()
pca.fit(data)
pc1 = pca.components_[0] * pca.explained_variance_[0] * 2
pc2 = pca.components_[1] * pca.explained_variance_[1] * 2
fow = pc1 + np.array([-0.4, 0.4])
fow = (fow / np.linalg.norm(fow)) * (pca.explained_variance_[0] * 0.7) * 2

cartax.scatter(data[:, 0], data[:, 1], s=25, color='lightgrey', edgecolor='white')
cartax.plot([0, pc1[0]], [0, pc1[1]], color='k', label=r'$\mathbf{PC}_{1}$')
pc1 = np.negative(pc1)
cartax.plot([0, pc1[0]], [0, pc1[1]], color='k')

cartax.plot([0, fow[0]], [0, fow[1]], lw=2, color=color.RAW, label=r'$\mathbf{\beta}_{1}$')
fow = np.negative(fow)
cartax.plot([0, fow[0]], [0, fow[1]], lw=2, color=color.RAW)

cartax.plot([0, pc2[0]], [0, pc2[1]], lw=2, color=color.CORRECTED, label=r'$\mathbf{\beta}_{2}$')
pc2 = np.negative(pc2)
cartax.plot([0, pc2[0]], [0, pc2[1]], lw=2, color=color.CORRECTED)

cartax.set_xlabel('Dim 1')
cartax.set_ylabel('Dim 2')

cartax.legend(frameon=False)

cartax.axis('square')

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()