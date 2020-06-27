"""
Calculate LV using difference of covariance matrices.
Show cov. in large / small, and difference. Show eigenspectrum raw
and unshuffled. Show fraction variance along each overall PC (as a reference frame)
and finally, show correlation (or lack of) with gain modulation.

Over all sites show:
    delta dprime depends on beta2 vs. dU
    lack of correlation between first order effects and second order effects
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES

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

savefig = False
fig_fn = PY_FIGURES_DIR + 'supp_ncLV.svg'

site = 'DRX006b.e65:128' #'BOL006b' #'DRX006b.e65:128'
batch = 289

fs = 4
ex_site = site
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


# ======================================== Load dprime results ================================================
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
sim1 = 'dprime_simInTDR_sim1_jk10_zscore_nclvz_fixtdr2'
sim2 = 'dprime_simInTDR_sim2_jk10_zscore_nclvz_fixtdr2'
estval = '_test'
cmap = 'PRGn'
high_var_only = False
all_sites = True

# where to crop the data
if estval == '_train':
    x_cut = (2, 9.5)
    y_cut = (0.05, .5) 
elif estval == '_test':
    x_cut = (1, 8)
    y_cut = (0.2, 1) 

if all_sites:
    sites = ALL_SITES
else:
    sites = HIGHR_SITES

df = []
df_sim1 = []
df_sim2 = []
path = DPRIME_DIR
for site in sites:
    if site in LOWR_SITES: mn = modelname.replace('_jk10', '_jk1_eev') 
    else: mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results

    if site in LOWR_SITES: mn = sim1.replace('_jk10', '_jk1_eev') 
    else: mn = sim1
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results_sim1 = loader.load_results(fn)
    _df_sim1 = results_sim1.numeric_results

    if site in LOWR_SITES: mn = sim2.replace('_jk10', '_jk1_eev') 
    else: mn = sim2
    fn = os.path.join(path, site, mn+'_TDR.pickle')
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

df_all = pd.concat(df)
df_sim1_all = pd.concat(df_sim1)
df_sim2_all = pd.concat(df_sim2)

# filter based on x_cut / y_cut
mask1 = (df_all['dU_mag'+estval] < x_cut[1]) & (df_all['dU_mag'+estval] > x_cut[0])
mask2 = (df_all['cos_dU_evec'+estval] < y_cut[1]) & (df_all['cos_dU_evec'+estval] > y_cut[0])
df = df_all[mask1 & mask2]
df_sim1 = df_sim1_all[mask1 & mask2]
df_sim2 = df_sim2_all[mask1 & mask2]

# append the simulation results as columns in the raw dataframe
df['sim1'] = df_sim1['state_diff']
df['sim2'] = df_sim2['state_diff']
df['2nd_residual'] = df['state_diff'] - df['sim1']

# load LV results for all sites
fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/nc_zscore_lvs.pickle'
with open(fn, 'rb') as handle:
    lv_dict = pickle.load(handle)

gb2 = np.abs([lv_dict[s]['b2_corr_gain'] for s in df.site.unique()])
pcb2 = np.abs([lv_dict[s]['b2_dot_pc1'] for s in df.site.unique()])
b2_tot_var = [lv_dict[s]['b2_tot_var_ratio'] for s in df.site.unique()]
b2_var_pc1_ratio = [lv_dict[s]['b2_var_pc1_ratio'] for s in df.site.unique()]

ex_gb2 = abs(lv_dict[site]['b2_corr_gain'])
ex_pcb2 = abs(lv_dict[site]['b2_dot_pc1'])

ex_var_pc1_ratio = lv_dict[ex_site]['b2_var_pc1_ratio']
ex_b2_tot_var = lv_dict[ex_site]['b2_tot_var_ratio']

# ==============================================================================================
# layout figure panels and plot

f = plt.figure(figsize=(12, 6))

lrgax = plt.subplot2grid((2, 4), (0, 1))
smax = plt.subplot2grid((2, 4), (0, 0))
difax = plt.subplot2grid((2, 4), (0, 2))
espec = plt.subplot2grid((2, 4), (1, 0))

#b1b2 = plt.subplot2grid((2, 5), (1, 0))
pcvar = plt.subplot2grid((2, 4), (1, 1), colspan=2)

varpc = plt.subplot2grid((2, 4), (0, 3)) 
#gvspc = plt.subplot2grid((2, 4), (0, 3))
#b2dU = plt.subplot2grid((2, 5), (1, 3))
dpax = plt.subplot2grid((2, 4), (1, 3))

vmin = -0.5
vmax = 0.5
lrgax.imshow(big, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
lrgax.set_title(r"$\Sigma_{large}$")
lrgax.set_xticks([], [])
lrgax.set_yticks([], [])
lrgax.set_xlabel('Units (sorted by depth)')
lrgax.set_ylabel('Units (sorted by depth)')

smax.imshow(small, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
smax.set_title(r"$\Sigma_{small}$")
smax.set_xticks([], [])
smax.set_yticks([], [])
smax.set_xlabel('Units (sorted by depth)')
smax.set_ylabel('Units (sorted by depth)')

difax.imshow(diff, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
difax.set_title(r"$\Sigma_{small} - \Sigma_{large}$")
difax.set_xticks([], [])
difax.set_yticks([], [])
difax.set_xlabel('Units (sorted by depth)')
difax.set_ylabel('Units (sorted by depth)')

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
espec.set_xticks(np.arange(x[0], x[-1], 10))
espec.set_xticklabels(np.arange(0, len(x), 10))

'''
b1b2.scatter(evecs[:, 0], g, color='grey', edgecolor='white', s=25)
b1b2.legend([r"$\rho = %s$" % np.round(np.corrcoef(g, evecs[:, 0])[0, 1], 2)], frameon=False)
b1b2.axhline(0, linestyle='--', color='k')
b1b2.axvline(0, linestyle='--', color='k')
b1b2.set_xlabel(r"$\beta_{2}$")
b1b2.set_ylabel("Gain Modulation")
b1b2.set_title('Per-neuron coefficients')
'''

ncells = resp_matrix.shape[0]
pcvar.bar(range(0, ncells), pca.explained_variance_ratio_, edgecolor='k', width=0.7, color='lightgrey', label='Total variance')
pcvar.bar(range(0, ncells), var, edgecolor='k', width=0.7, color='tab:orange', label="2nd-order variance")
pcvar.set_ylabel('Explained variance \n ratio')
pcvar.set_xlabel('PC')
pcvar.legend(frameon=False)
pcvar.set_title("Variance explained by second-order dimension \n along each principal component")


# plot ratio of beta2 variance : pc1 variance vs. alignment with PC1
sig_beta2 = [s for s in lv_dict.keys() if lv_dict[s]['beta2_sig']]
lowr_mask = np.array([True if (s in LOWR_SITES) and (s in sig_beta2) else False for s in df.site.unique()])
highr_mask = np.array([True if (s in HIGHR_SITES) and (s in sig_beta2) else False for s in df.site.unique()])
varpc.scatter(np.array(b2_tot_var)[lowr_mask], np.array(b2_var_pc1_ratio)[lowr_mask], color='grey', marker='D', edgecolor='white', s=30)
varpc.scatter(np.array(b2_tot_var)[highr_mask], np.array(b2_var_pc1_ratio)[highr_mask], color='k', edgecolor='white', s=50)
varpc.scatter(ex_b2_tot_var, ex_var_pc1_ratio, color='tab:orange', edgecolor='white', s=50, zorder=3)
varpc.axhline(0, linestyle='--', color='grey')
varpc.axvline(0, linestyle='--', color='grey')
varpc.set_xlabel(r"$\frac{var(\beta_2)}{var_{total}}$")
varpc.set_ylabel(r"$\frac{var(\beta_2)}{var(PC_1)}$")
varpc.set_xlim((-0.01, None))
varpc.set_ylim((-0.1, 1))

#m1 = df['beta2_dot_dU']>0.3
#m2 = df['beta2_dot_dU']<0.2
#low = df[m2].groupby(by='site').mean()['sim2']
#high = df[m1].groupby(by='site').mean()['sim2']

# split by up/low quartile for each site independently (or by median...)
df['high_mask'] = False
df['low_mask'] = False
for s in sig_beta2:
    qts = df[df.site==s].beta2_dot_dU.quantile([0.25, 0.5, 0.75])
    upper = qts.iloc[1]
    lower = qts.iloc[1]
    mh = (df.site==s) & (df['beta2_dot_dU'] >= upper)
    ml = (df.site==s) & (df['beta2_dot_dU'] <= lower)
    df.loc[mh, 'high_mask'] = True
    df.loc[ml, 'low_mask'] = True
low = df[df.low_mask].groupby(by='site').mean()['2nd_residual']
high = df[df.high_mask].groupby(by='site').mean()['2nd_residual']

try:
    dpax.scatter([np.zeros(low.loc[low.index.isin(LOWR_SITES)].shape[0]), 
            np.ones(low.loc[low.index.isin(LOWR_SITES)].shape[0])],
          [low.loc[low.index.isin(LOWR_SITES)], high.loc[high.index.isin(LOWR_SITES)]], marker='D', color='grey', s=30, edgecolor='white', zorder=2)
except:
    pass
dpax.scatter([np.zeros(low.loc[low.index.isin(HIGHR_SITES)].shape[0]), 
            np.ones(low.loc[low.index.isin(HIGHR_SITES)].shape[0])],
          [low.loc[low.index.isin(HIGHR_SITES)], high.loc[high.index.isin(HIGHR_SITES)]], marker='o', color='k', s=50, edgecolor='white', zorder=3)

dpax.scatter([0, 1], [low.loc[ex_site], high.loc[ex_site]], marker='o', color='tab:orange', edgecolor='white', zorder=4)

for l, h, s in zip(low.values, high.values, high.index):
    if s == ex_site:
        dpax.plot([0, 1], [l, h], 'tab:orange', zorder=3, lw=2)
    elif s in LOWR_SITES:
        dpax.plot([0, 1], [l, h], 'grey', zorder=1)
    else:
        dpax.plot([0, 1], [l, h], 'k', zorder=2)
    
dpax.axhline(0, linestyle='--', color='k')
dpax.set_xticks([0, 1])
dpax.set_xticklabels(['Low', 'High'])
dpax.set_xlabel(r"$\beta_2$ vs. $\Delta \mu$ similarity")
dpax.set_ylabel(r"$\Delta d'$ (2nd-order)")
dpax.set_xlim((-1, 2))

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()