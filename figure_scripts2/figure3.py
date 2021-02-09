"""
Example "confusion" matrix plotting dprime / population responses for all
stimuli at a site. Highlight that changes are not explained by first order 
response properties (e.g. PC_1 response variance)

*can* predict some of the delta dprime with other features 
    (noise alignement and dU mag)
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR2, CACHE_PATH
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES
from regression_helper import fit_OLS_model
import figure_scripts2.helper as chelp

import charlieTools.nat_sounds_ms.preprocessing as nat_preproc
import charlieTools.nat_sounds_ms.decoding as decoding

import os
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

modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
recache = False
site = 'DRX008b.e65:128' #'DRX007a.e65:128' #'DRX008b.e65:128' #'DRX007a.e65:128'
batch = 289
duration = 1   # length of stimuli (for making spectrograms)
prestim = 0.25 # legnth of stimuli (for making spectrograms)
soundpath = '/auto/users/hellerc/code/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set4/'

# get decoding results
loader = decoding.DecodingResults()
fn = os.path.join(DPRIME_DIR, site, modelname+'_TDR.pickle')
results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
df = results.numeric_results.loc[results.evoked_stimulus_pairs]
df['noiseAlign'] = results.slice_array_results('cos_dU_evec_test', results.evoked_stimulus_pairs, 2, idx=(0, 0))[0]

X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, return_epoch_list=True)
ncells = X.shape[0]
nreps = X.shape[1]
nstim = X.shape[2]
nbins = X.shape[3]
sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
nstim = nstim * nbins

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
f = plt.figure(figsize=(6, 6))

bp = plt.subplot2grid((2, 2), (0, 0))
sp = plt.subplot2grid((2, 2), (0, 1))
diff = plt.subplot2grid((2, 2), (1, 0))
reg = plt.subplot2grid((2, 2), (1, 1))

# get big pupil / small pupil projected response, scale the same way to put between 0 / 1
bpsp_proj = proj[:, :, :2]
ma = bpsp_proj.max()
mi = bpsp_proj[:, :, :2].min()
ran = ma - mi
bpsp_proj += abs(mi)
bpsp_proj /= ran

# plot big pupil
bp_proj = np.stack([bpsp_proj[i, pup_mask[0, :, i], :] for i in range(pup_mask.shape[-1])])
chelp.plot_confusion_matrix(df, 
                    metric='bp_dp',
                    spectrogram=np.sqrt(stimulus)**(1/2),
                    resp_fs=4,
                    stim_fs=stim_fs,
                    pcs = bp_proj,
                    vmin=0,
                    vmax=100,
                    ax=bp
                    )
bp.set_title(r"Large pupil $d'^2$")
# plot small pupil
# plot big pupil
sp_proj = np.stack([bpsp_proj[i, ~pup_mask[0, :, i], :] for i in range(pup_mask.shape[-1])])
chelp.plot_confusion_matrix(df, 
                    metric='sp_dp',
                    spectrogram=np.sqrt(stimulus)**(1/2),
                    resp_fs=4,
                    stim_fs=stim_fs,
                    pcs = sp_proj,
                    vmin=0,
                    vmax=100,
                    ax=sp
                    )
sp.set_title(r"Small pupil $d'^2$")

# plot difference
df['delta'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
chelp.plot_confusion_matrix(df, 
                    metric='delta',
                    spectrogram=np.sqrt(stimulus)**(1/2),
                    resp_fs=4,
                    stim_fs=stim_fs,
                    ax=diff
                    )
diff.set_title(r"$\Delta d'^2$")

f.tight_layout()

plt.show()