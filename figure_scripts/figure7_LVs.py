"""
Calculate LV using difference of covariance matrices.
Show cov. in large / small, and difference. Show eigenspectrum raw
and unshuffled. Show fraction variance along each overall PC (as a reference frame)
and finally, show correlation (or lack of) with gain modulation.

Over all sites show:
    delta dprime depends on beta2 vs. dU
    lack of correlation between first order effects and second order effects
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import pandas as pd
import scipy.stats as ss
import os

from charlieTools.preprocessing import generate_state_corrected_psth, bandpass_filter_resp, sliding_window
import charlieTools.nat_sounds_ms.decoding as decoding
import charlieTools.nat_sounds_ms.preprocessing as preproc
import charlieTools.nat_sounds_ms.dim_reduction as dr
import charlieTools.preprocessing as cpreproc

from nems_lbhb.baphy import parse_cellid
from nems_lbhb.preprocessing import create_pupil_mask
np.random.seed(123)

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

savefig = True
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig7_LV.svg'

site = 'TAR010c' #'BOL006b' #'DRX006b.e65:128'
batch = 289

fs = 4
ops = {'batch': batch, 'cellid': site}
xmodel = 'ns.fs{}.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'.format(fs)
if batch == 294:
    xmodel = xmodel.replace('ns.fs4.pup', 'ns.fs4.pup.voc')
path = '/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/'
low = 0.5
high = 2  # for filtering the projection

cells, _ = parse_cellid(ops)
rec = generate_state_corrected_psth(batch=batch, modelname=xmodel, cellids=cells, siteid=site,
                                    cache_path=path, gain_only=False, recache=False)
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

real_dict_small = cpreproc.zscore_per_stim(real_dict_small, d2=real_dict_small, with_std=True)
real_dict_big = cpreproc.zscore_per_stim(real_dict_big, d2=real_dict_big, with_std=True)
real_dict_all = cpreproc.zscore_per_stim(real_dict_all, d2=real_dict_all, with_std=True)
pred_dict_all = cpreproc.zscore_per_stim(pred_dict_all, d2=pred_dict_all, with_std=True)

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


# ====================================== load first order results =============================
path = '/auto/users/hellerc/results/nat_pupil_ms/first_order_model_results/'
df = pd.concat([pd.read_csv(os.path.join(path,'d_289_pup_sdexp.csv'), index_col=0),
                pd.read_csv(os.path.join(path,'d_294_pup_sdexp.csv'), index_col=0)])
df = df[df.state_chan=='pupil'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se']) 
gain = pd.DataFrame(df.loc[:, pd.IndexSlice['gain_mod', 'st.pup']])
gain.loc[:, 'site'] = [i.split('-')[0] for i in gain.index]
gain = gain.loc[[c for c in rec['resp'].chans]]
g = gain['gain_mod']['st.pup'].values
g = [g for g in g]


# load overall LV results
fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/nc_zscore_lvs.pickle'
with open(fn, 'rb') as handle:
    lv_dict = pickle.load(handle)

gb2 = np.abs([lv_dict[s]['b2_corr_gain'] for s in lv_dict.keys()])
pcb2 = np.abs([lv_dict[s]['b2_dot_pc1'] for s in lv_dict.keys()])

ex_gb2 = abs(lv_dict[site]['b2_corr_gain'])
ex_pcb2 = abs(lv_dict[site]['b2_dot_pc1'])

# ======================================== Load dprime results ================================================
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
sim1 = 'dprime_sim1_jk10_zscore_nclvz_fixtdr2'
sim2 = 'dprime_sim2_jk10_zscore_nclvz_fixtdr2'
estval = '_test'
nbins = 20
cmap = 'PRGn'
high_var_only = True

# where to crop the data
if estval == '_train':
    x_cut = (2, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = (1, 8)
    y_cut = (0.2, 1) 

sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
        'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
        'DRX007a.e1:64', 'DRX007a.e65:128', 
        'DRX008b.e1:64', 'DRX008b.e65:128']

df = []
df_sim1 = []
df_sim2 = []
for site in sites:
    path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
    fn = os.path.join(path, site, modelname+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results

    fn = os.path.join(path, site, sim1+'_TDR.pickle')
    results_sim1 = loader.load_results(fn)
    _df_sim1 = results_sim1.numeric_results

    fn = os.path.join(path, site, sim2+'_TDR.pickle')
    results_sim2 = loader.load_results(fn)
    _df_sim2 = results_sim2.numeric_results

    stim = results.evoked_stimulus_pairs
    high_var_pairs = pd.read_csv('/auto/users/hellerc/results/nat_pupil_ms/dprime_new/high_pvar_stim_combos.csv', index_col=0)
    high_var_pairs = high_var_pairs[high_var_pairs.site==site].index.get_level_values('combo')
    if high_var_only:
        stim = [s for s in stim if s in high_var_pairs]
    if len(stim) == 0:
        pass
    else:
        _df = _df.loc[pd.IndexSlice[stim, 2], :]
        _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
        _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
        _df['state_diff'] = (_df['bp_dp'] - _df['sp_dp']) / _df['dp_opt_test']
        _df['site'] = site
        df.append(_df)

        _df_sim1 = _df_sim1.loc[pd.IndexSlice[stim, 2], :]
        _df_sim1['state_diff'] = (_df_sim1['bp_dp'] - _df_sim1['sp_dp']) / _df['dp_opt_test']
        _df_sim1['site'] = site
        df_sim1.append(_df_sim1)

        _df_sim2 = _df_sim2.loc[pd.IndexSlice[stim, 2], :]
        _df_sim2['state_diff'] = (_df_sim2['bp_dp'] - _df_sim2['sp_dp']) / _df['dp_opt_test']
        _df_sim2['site'] = site
        df_sim2.append(_df_sim2)

df = pd.concat(df)
df_sim1 = pd.concat(df_sim1)
df_sim2 = pd.concat(df_sim2)

# filter based on x_cut / y_cut
mask1 = (df['dU_mag'+estval] < x_cut[1]) & (df['dU_mag'+estval] > x_cut[0])
mask2 = (df['cos_dU_evec'+estval] < y_cut[1]) & (df['cos_dU_evec'+estval] > y_cut[0])
df = df[mask1 & mask2]
df_sim1 = df_sim1[mask1 & mask2]
df_sim2 = df_sim2[mask1 & mask2]

# append the simulation results as columns in the raw dataframe
df['sim1'] = df_sim1['state_diff']
df['sim2'] = df_sim2['state_diff']
df['dU_diff'] = (df['bp_dU_mag'] - df['sp_dU_mag']) / df['dU_mag_test']

# ==============================================================================================
# layout figure panels and plot

f = plt.figure(figsize=(15, 6))

lrgax = plt.subplot2grid((2, 5), (0, 1))
smax = plt.subplot2grid((2, 5), (0, 0))
difax = plt.subplot2grid((2, 5), (0, 2))
espec = plt.subplot2grid((2, 5), (0, 3))

b1b2 = plt.subplot2grid((2, 5), (1, 0))
pcvar = plt.subplot2grid((2, 5), (1, 1), colspan=2)

gvspc = plt.subplot2grid((2, 5), (0, 4))
b2dU = plt.subplot2grid((2, 5), (1, 3))
dpax = plt.subplot2grid((2, 5), (1, 4))

vmin = -0.75
vmax = 0.75
lrgax.imshow(big, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
lrgax.set_title(r"$\Sigma_{large}$")

smax.imshow(small, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
smax.set_title(r"$\Sigma_{small}$")

difax.imshow(diff, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
difax.set_title(r"$\Sigma_{small} - \Sigma_{large}$")

x = np.arange(0, len(evals)) - int(len(evals)/2)
espec.plot(x, shuf_evals, '.-', label='pupil shuffle')
espec.plot(x, evals, '.-', label='raw data')
espec.axhline(0, linestyle='--', color='grey')
espec.axvline(0, linestyle='--', color='grey')
espec.set_title("Eigenspectrum of "
                    "\n"
                    r"$\Sigma_{small} - \Sigma_{large}$")
espec.set_ylabel(r"$\lambda$")
espec.set_xlabel(r"$\alpha$")
espec.legend(frameon=False)

b1b2.scatter(evecs[:, 0], g, color='grey', edgecolor='white', s=25)
b1b2.legend([r"$\rho = %s$" % np.round(np.corrcoef(g, evecs[:, 0])[0, 1], 2)], frameon=False)
b1b2.axhline(0, linestyle='--', color='k')
b1b2.axvline(0, linestyle='--', color='k')
b1b2.set_xlabel(r"$\beta_{2}$")
b1b2.set_ylabel("Gain Modulation")
b1b2.set_title('Per-neuron coefficients')

ncells = resp_matrix.shape[0]
pcvar.bar(range(0, ncells), pca.explained_variance_ratio_, edgecolor='k', width=0.7, color='lightgrey', label='Raw data')
pcvar.bar(range(0, ncells), var, edgecolor='k', width=0.7, color='tab:orange', label="2nd-order variance")
pcvar.set_ylabel('Explained \n variance')
pcvar.set_xlabel('PC')
pcvar.legend(frameon=False)
pcvar.set_title("Variance explained by second order dimension " + r"($\mathbf{e}_1 \cdot \beta_{2}$)")

gvspc.scatter(gb2, pcb2, color='grey', edgecolor='white', s=35)
gvspc.scatter(ex_gb2, ex_pcb2, color='tab:orange', edgecolor='white', s=50)
gvspc.axhline(0, linestyle='--', color='k')
gvspc.axvline(0, linestyle='--', color='k')
gvspc.set_ylim((-0.1, 1))
gvspc.set_xlim((-0.1, 1))
gvspc.set_xlabel(r"$corr(gain, \beta_{2})$")
gvspc.set_ylabel(r"$\mathbf{e}_{1} \cdot \beta_{2}$")

b2dU.hist(df['beta2_dot_dU'], edgecolor='k', 
            color='lightgrey', rwidth=0.7, bins=np.arange(0, 1, 0.05))
b2dU.set_title("Second-order \n overlap with signal")
b2dU.set_xlabel(r"$\beta_2 \cdot \Delta \mu$")
b2dU.set_ylabel('per Recording Site, \n Stimulus Pair')

m1 = df['beta2_dot_dU']>0.3
m2 = df['beta2_dot_dU']<0.2
low = df[m2].groupby(by='site').mean()['sim2']
high = df[m1].groupby(by='site').mean()['sim2']

dpax.plot([np.zeros(low.shape[0]), 
            np.ones(low.shape[0])],
          [low, high], 'o', color='grey', zorder=2)
dpax.plot([0, 1], [low.loc['TAR010c'], high.loc['TAR010c']], 'o', color='tab:orange', zorder=3)

for l, h, s in zip(low.values, high.values, high.index):
    if s == 'TAR010c':
        dpax.plot([0, 1], [l, h], 'tab:orange', zorder=2, lw=2)
    else:
        dpax.plot([0, 1], [l, h], 'k-', zorder=1)
    

dpax.set_xticks([0, 1])
dpax.set_xticklabels(['Low', 'High'])
dpax.set_xlabel('Second-order overlap with signal')
dpax.set_ylabel(r"$\Delta d'$ (2nd-order)")
dpax.set_xlim((-1, 2))

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()