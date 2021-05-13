"""
Example "confusion" matrix plotting dprime / population responses for all
stimuli at a site. Highlight that changes are not explained by first order 
response properties (e.g. PC_1 response variance)

Also, show big /small pupil d-pdrime scatter plot
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR2, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, CPN_SITES
from regression_helper import fit_OLS_model
import figure_scripts2.helper as chelp

import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding
from charlieTools.statistics import get_direct_prob, get_bootstrapped_sample

import os
import pandas as pd
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from itertools import combinations
from sklearn.decomposition import PCA
from scipy.io import wavfile
import nems.analysis.gammatone.gtgram as gt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

savefig = False
fig_fn = PY_FIGURES_DIR2 + 'fig3.svg'

modelname = 'dprime_jk10_zscore_nclvz_fixtdr2' #_model-LV-psth.fs4.pup-loadpred-st.pup.pvp-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5.pc1' #
#modelname = "dprime_jk10_zscore_nclvz_fixtdr2_model-LV-psth.fs4.pup-loadpred-st.pup0.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.2xR.d-inoise.3xR_ccnorm.t5.ss3"
recache = False
site = 'DRX008b.e65:128' #'DRX007a.e65:128' #'DRX008b.e65:128' #'DRX007a.e65:128'
batch = 289
duration = 1   # length of stimuli (for making spectrograms)
prestim = 0.25 # legnth of stimuli (for making spectrograms)
pc_div  = 8 # how much space to leave around conf. matrix edge for PC (bigger this is, the less space. Default = 16)
soundpath = '/auto/users/hellerc/code/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set4/'

# get decoding results
loader = decoding.DecodingResults()
fn = os.path.join(DPRIME_DIR, str(batch), site, modelname+'_TDR.pickle')
results = loader.load_results(fn, cache_path=None, recache=recache)
df = results.numeric_results.loc[results.evoked_stimulus_pairs]
df['noiseAlign'] = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, 2, idx=(0, 0))[0]

X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)
ncells = X.shape[0]
nreps = X.shape[1]
nstim = X.shape[2]
nbins = X.shape[3]
sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
nstim = nstim * nbins

# ============================= LOAD DPRIME =========================================
path = DPRIME_DIR
loader = decoding.DecodingResults()
n_components = 2
recache = False
df_all = []
sites = CPN_SITES
batches = [331]*len(CPN_SITES)
modelname = modelname.replace('loadpred', 'loadpred.cpn')

#sites = HIGHR_SITES
#batches = [289] * len(sites)

for batch, site in zip(batches, sites):
    if (site in LOWR_SITES) & (batch != 331):
        mn = modelname.replace('_jk10', '_jk1_eev')
    else:
        mn = modelname
    if site in ['BOL005c', 'BOL006b']:
        batch = 294
    try:
        fn = os.path.join(path, str(batch), site, mn+'_TDR.pickle')
        results = loader.load_results(fn, cache_path=None, recache=recache)
        _df = results.numeric_results
    except:
        print(f"WARNING!! NOT LOADING SITE {site}")

    stim = results.evoked_stimulus_pairs
    _df = _df.loc[pd.IndexSlice[stim, 2], :]
    _df['site'] = site
    df_all.append(_df)

df_all = pd.concat(df_all)

# =========================== generate a list of stim pairs ==========================
# these are the indices of the decoding results dataframes
all_combos = list(combinations(range(nstim), 2))
spont_bins = np.argwhere(sp_bins[0, 0, :])
spont_combos = [c for c in all_combos if (c[0] in spont_bins) & (c[1] in spont_bins)]
ev_ev_combos = [c for c in all_combos if (c[0] not in spont_bins) & (c[1] not in spont_bins)]
spont_ev_combos = [c for c in all_combos if (c not in ev_ev_combos) & (c not in spont_combos)]

X = X.reshape(ncells, nreps, nstim)
pup_mask = pup_mask.reshape(1, nreps, nstim)
ev_bins = list(set(range(X.shape[-1])).difference(set(spont_bins.squeeze())))
Xev = X[:, :, ev_bins]

# ============================= DO PCA ================================
Xu = Xev.mean(axis=1)
spont = X[:, :, spont_bins.squeeze()].mean(axis=1).mean(axis=-1, keepdims=True)
Xu_center = Xu - spont # subtract spont
pca = PCA()
pca.fit(Xu_center.T)

spont = spont[:, :, np.newaxis] # for subtracting from single trial data
X_spont = X - spont
proj = (X_spont).T.dot(pca.components_.T)

# ================== BUILD SPECTROGRAM FOR FIGURE ======================
# (high res)
stimulus = []
for i, epoch in enumerate(epochs):
    print(f"Building spectrogram {i} / {len(epochs)} for epoch: {epoch}")
    soundfile = soundpath + epoch.strip('STIM_')
    fs, data = wavfile.read(soundfile)
    # pad / crop data
    data = data[:int(duration * fs)]
    spbins = int(prestim * fs)
    data = np.concatenate((np.zeros(spbins), data))
    spec = gt.gtgram(data, fs, 0.004, 0.002, 100, 0)
    stimulus.append(spec)
stimulus = np.concatenate(stimulus, axis=-1)
stim_fs = 500

# ====================== LOAD REGRESSION ANALYSIS ACROSS SITES ==========================
# for example site alone
df['r1'] = [proj[int(idx.split('_')[0]), :, 0].mean() for idx in df.index.get_level_values(0)]
df['r2'] = [proj[int(idx.split('_')[1]), :, 0].mean() for idx in df.index.get_level_values(0)]
y = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
X = df[['r1', 'r2', 'dU_mag_test', 'noiseAlign']]
X['interaction'] = X['r1'] * X['r2']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()


# ================================== BUILD FIGURE =======================================
f = plt.figure(figsize=(13, 6))

gs = mpl.gridspec.GridSpec(2, 12, width_ratios=np.ones(12), height_ratios=[1, 0.05],
         wspace=0.0, hspace=0.0, top=0.9, bottom=0.1, left=0.0, right=1.0)
bp = f.add_subplot(gs[0, 0:3])
sp = f.add_subplot(gs[0, 3:6])
diff = f.add_subplot(gs[0, 6:9])
scax = f.add_subplot(gs[0, 9:])

# get big pupil / small pupil projected response, scale the same way to put between 0 / 1
bpsp_proj = proj[:, :, :2].copy()
ma = bpsp_proj.max()
mi = bpsp_proj[:, :, :2].min()
ran = ma - mi
bpsp_proj += abs(mi)
bpsp_proj /= ran
baseline = abs(mi)
baseline /= ran

nbins = int(((stimulus.shape[1]/stim_fs) * 4) / len(epochs))
df['delta'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
# plot big pupil
bp_proj = np.stack([bpsp_proj[i, pup_mask[0, :, i], :] for i in range(pup_mask.shape[-1])])
im = chelp.plot_confusion_matrix(df, 
                    metric='bp_dp',
                    spectrogram=np.sqrt(stimulus)**(1/2),
                    sortby=('delta', nbins),
                    sort_method='1D',
                    resp_fs=4,
                    stim_fs=stim_fs,
                    pcs = bp_proj,
                    baseline=baseline,
                    vmin=0,
                    vmax=100,
                    pc_div=8,
                    ax=bp
                    )
bp.set_title(r"Large pupil $d'^2$")
cax = plt.subplot(gs[1, 1])
f.colorbar(im, ax=bp, cax=cax, orientation='horizontal', ticks=[0, 50, 100])
# plot small pupil
sp_proj = np.stack([bpsp_proj[i, ~pup_mask[0, :, i], :] for i in range(pup_mask.shape[-1])])
im = chelp.plot_confusion_matrix(df, 
                    metric='sp_dp',
                    spectrogram=np.sqrt(stimulus)**(1/2),
                    sortby=('delta', nbins),
                    sort_method='1D',
                    resp_fs=4,
                    stim_fs=stim_fs,
                    pcs = sp_proj,
                    baseline=baseline,
                    vmin=0,
                    vmax=100,
                    pc_div=pc_div,
                    ax=sp
                    )
sp.set_title(r"Small pupil $d'^2$")
cax = plt.subplot(gs[1, 4])
f.colorbar(im, ax=sp, cax=cax, orientation='horizontal', ticks=[0, 50, 100])

# plot difference
im = chelp.plot_confusion_matrix(df, 
                    metric='delta',
                    spectrogram=np.sqrt(stimulus)**(1/2),
                    sortby=('delta', nbins),
                    sort_method='1D',
                    resp_fs=4,
                    stim_fs=stim_fs,
                    vmin=-1,
                    vmax=1,
                    pc_div=pc_div,
                    ax=diff
                    )
diff.set_title(r"$\Delta d'^2$")
#cax = f.add_axes([0.1, 0.1, 0.1, 0.05])
cax = plt.subplot(gs[1, 7])
cbar = f.colorbar(im, ax=diff, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])

# plot scatter plot of delta dprime results
# plot dprime results
nSamples = 2000
idx = df_all[['bp_dp', 'sp_dp']].max(axis=1) < 100
nSamples = idx.sum()
sidx = np.random.choice(range(idx.sum()), nSamples, replace=False)
bp = df_all['bp_dp'].values[idx][sidx]
sp = df_all['sp_dp'].values[idx][sidx]
s = 5
xy = np.vstack([bp, sp])
z = gaussian_kde(xy)(xy)
scax.scatter(sp, bp, s=s, c=z, cmap='inferno')
scax.plot([0, 100], [0, 100], 'k--')
scax.set_xlabel("Small pupil")
scax.set_ylabel("Large pupil")
scax.set_title(r"Stimulus discriminability ($d'^2$)")
scax.axis('square')

# get statistics for all data
df_all['delta'] = (df_all['bp_dp'] - df_all['sp_dp']) #/ (df_all['bp_dp'] + df_all['sp_dp'])
d = {s: df_all[df_all.site==s]['delta'].values for s in df_all.site.unique()}
bs = get_bootstrapped_sample(d, even_sample=False, nboot=1000)
p = get_direct_prob(bs, np.zeros(bs.shape[0]))[0]

print(f"mean large pupil d': {df_all['bp_dp'].mean()}, {df_all['bp_dp'].sem()}")
print(f"mean small pupil d': {df_all['sp_dp'].mean()}, {df_all['sp_dp'].sem()}")
print(f"pval (bootstrapped): {p}")
print(f"Mean n stimulus pairs per session: {np.mean([len(d[s]) for s in d.keys()])}, {np.std([len(d[s]) for s in d.keys()]) / np.sqrt(len(d.keys()))}")


frac = []
for s in d.keys():
    frac.append(np.sum(d[s]<0) / len(d[s]))
print(f"Fraction of stimlulus pairs with decreases per site: {np.mean(frac)}, sem: {np.std(frac)/len(frac)}")


#f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()