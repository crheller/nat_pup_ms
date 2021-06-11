"""
1 - Stim PC1/2 ellipse plots: overall, big, and small pupil
    - one pair gets better with pupil, one gets worse
? - add schematic of dprime calculation onto the ellipse plot?
? - "key" that shows which sound sprectrogram corresponds to each ellipse
        * modeled off fig1 - sounds excerpts divided into bins. e.g. a_1 is excerpt a, bin 1
"""
from global_settings import HIGHR_SITES, LOWR_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR3, CACHE_PATH
import charlieTools.plotting as cplt
import charlieTools.nat_sounds_ms.decoding as decoding
import nems_lbhb.baphy as nb
from nems_lbhb.plots import plot_weights_64D
import nems_lbhb.preprocessing as preproc
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sf
from scipy.io import wavfile
import nems.analysis.gammatone.gtgram as gt
from itertools import combinations
from sklearn.decomposition import PCA
import pandas as pd
import os
import copy
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

savefig = False
fig_fn = PY_FIGURES_DIR3 + 'fig2.svg'
recache = False

site = 'AMT026a'
batch = 331

# ========= Load decoding results and pick stimuli ========
decoder = 'dprime_jk10_zscore_nclvz_fixtdr2-fa'
loader = decoding.DecodingResults()
fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+'_TDR.pickle')
res = loader.load_results(fn, cache_path=None, recache=recache)
ndim = 2
df = res.numeric_results.loc[pd.IndexSlice[res.evoked_stimulus_pairs, ndim], :]
df['delta_dprime'] = (df['bp_dp'] - df['sp_dp']) / (df['bp_dp'] + df['sp_dp'])
df['raw_delta'] = (df['bp_dp'] - df['sp_dp'])
stims = (0, 6, 9)

# ====================== Get PC data ======================
xf_model = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.so-inoise.2xR_ccnorm.t5.ss1"
xf_model = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss3"

X_raw, sp_bins, X_pup, pup_mask, epochs, spont_raw = decoding.load_site(site=site, 
                                                        batch=batch, 
                                                        xforms_modelname=None,
                                                        special=True)
if xf_model is not None:
    X, _, _, _, _, _ = decoding.load_site(site=site, 
                                                            batch=batch, 
                                                            xforms_modelname=xf_model,
                                                            special=True)
else:
    X = X_raw.copy()
ncells = X_raw.shape[0]
nreps = X_raw.shape[1]
nstim = X_raw.shape[2]
nbins = X_raw.shape[3]
sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
nstim = nstim * nbins

X = X[:, :X_raw.shape[1], :, :]

# =========================== generate a list of stim pairs ==========================
# these are the indices of the decoding results dataframes
all_combos = list(combinations(range(nstim), 2))
spont_bins = np.argwhere(sp_bins[0, 0, :])
spont_combos = [c for c in all_combos if (c[0] in spont_bins) & (c[1] in spont_bins)]
ev_ev_combos = [c for c in all_combos if (c[0] not in spont_bins) & (c[1] not in spont_bins)]
spont_ev_combos = [c for c in all_combos if (c not in ev_ev_combos) & (c not in spont_combos)]

X_raw = X_raw.reshape(ncells, nreps, nstim)
X = X.reshape(ncells, nreps, nstim)
pup_mask = pup_mask.reshape(1, nreps, nstim).squeeze()
ev_bins = list(set(range(X.shape[-1])).difference(set(spont_bins.squeeze())))
Xev = X[:, :, ev_bins]
Xev_raw = X_raw[:, :, ev_bins]
# ============================= DO PCA ================================
# use raw data for PCs
Xu = Xev_raw.mean(axis=1)
if batch==331:
    spont = np.tile(spont_raw, [Xu.shape[-1], 1]).T
else:
    spont = X_raw[:, :, spont_bins.squeeze()].mean(axis=1).mean(axis=-1, keepdims=True)
Xu_center = Xu - spont # subtract spont
pca = PCA()
pca.fit(Xu_center.T)

spont = spont[:, :, np.newaxis] # for subtracting from single trial data
if batch==331:
    X_spont = X - spont.transpose(0, 2, 1)
else:
    X_spont = X - spont
proj = (X_spont).T.dot(pca.components_[0:2, :].T)

pr1 = proj[stims[0]]
pr2 = proj[stims[1]]
pr3 = proj[stims[2]]

# ========================= GET SPECTROGRAMS ===========================
# for each epoch, then highlight the example bins
spectrogram = {}
for alnum, epoch in zip(list(map(chr, np.arange(97, 97+len(epochs)))), epochs):
    ep_name = epoch.strip('STIM_probe:')
    soundfile = f'/auto/users/hellerc/code/baphy/Config/lbhb/SoundObjects/@NaturalPairs/NatPairSounds/{ep_name}.wav'
    # spectrogram
    fs, data = wavfile.read(soundfile)
    # pad / crop data
    data = data[int(0.25 * fs):int(1 * fs)]
    spbins = 0
    postbins = 0
    data = np.concatenate((np.zeros(spbins), data, np.zeros(postbins)))
    spec = gt.gtgram(data, fs, 0.01, 0.002, 100, 0)
    spectrogram[epoch] = {}
    spectrogram[epoch]['key'] = alnum
    spectrogram[epoch]['data'] = spec

# ============================== Make figure =========================================
f = plt.figure(figsize=(16, 4))

el_all = plt.subplot2grid((4, 4), (0, 1), rowspan=4)
el_bp = plt.subplot2grid((4, 4), (0, 2), rowspan=4)
el_sp = plt.subplot2grid((4, 4), (0, 3), rowspan=4)

# plot spectrograms
cols = plt.cm.get_cmap('tab10')
mappers = np.unique(np.array([x for x in res.mapping.values()]).flatten()) 
for i, (ep, spec) in enumerate(spectrogram.items()):
    ax = plt.subplot2grid((4, 4), (i, 0))
    ax.imshow(np.sqrt(spec['data']), origin='lower', cmap='Greys')
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(spec['key'], rotation=0)
    if i == 0:
        ax.set_title("Sound excerpt")

    # find if this corresponds to any of the selected stimuli
    for j, s in enumerate(stims):
        try:
            ek = res.mapping[f'{s}_{s+1}']
            ek = ek[0]
        except:
            ek = res.mapping[f'{s-1}_{s}']
            ek = ek[1]
        if '_'.join(ek.split('_')[:-1]) == ep:
            b = int(ek.split('_')[-1])
            st = (spec['data'].shape[-1] / 3) * b
            en = (spec['data'].shape[-1] / 3) * b + (spec['data'].shape[-1] / 3) * (b + 1)
            ax.axvline(st, color=cols(j), lw=1)
            ax.axvline(en, color=cols(j), lw=1)
            yl = ax.get_ylim()[-1]
            yll = ax.get_ylim()[0]
            ax.plot([st, en], [yl, yl], color=cols(j))
            ax.plot([st, en], [yll, yll], color=cols(j))
ax.set_xlabel("Time (ms)")
ax.set_xticks(np.linspace(0, spec['data'].shape[-1], 4))
ax.set_xticklabels([0, 250, 500, 750])

# plot pc ellipse plots
for i in range(proj.shape[0]):
    if i in ev_bins:
        el = cplt.compute_ellipse(proj[i, :, 0], proj[i, :, 1])
        el_all.plot(el[0], el[1], color='lightgrey', lw=0.5, zorder=-1)
        bp = pup_mask[:, i]
        el = cplt.compute_ellipse(proj[i, bp, 0], proj[i, bp, 1])
        el_bp.plot(el[0], el[1], color='lightgrey', lw=0.5, zorder=-1)
        el = cplt.compute_ellipse(proj[i, ~bp, 0], proj[i, ~bp, 1])
        el_sp.plot(el[0], el[1], color='lightgrey', lw=0.5, zorder=-1)

ms = 2
lw = 1
el1 = cplt.compute_ellipse(pr1[:, 0], pr1[:, 1])
el_all.plot(el1[0], el1[1], color='tab:blue', lw=lw)
el_all.scatter(pr1[:, 0], pr1[:, 1], color='tab:blue', s=ms)
bp = pup_mask[:, stims[0]]
el1 = cplt.compute_ellipse(pr1[bp, 0], pr1[bp, 1])
el_bp.plot(el1[0], el1[1], color='tab:blue', lw=lw)
el1 = cplt.compute_ellipse(pr1[~bp, 0], pr1[~bp, 1])
el_sp.plot(el1[0], el1[1], color='tab:blue', lw=lw)

el2 = cplt.compute_ellipse(pr2[:, 0], pr2[:, 1])
el_all.plot(el2[0], el2[1], color='tab:orange', lw=lw)
el_all.scatter(pr2[:, 0], pr2[:, 1], color='tab:orange', s=ms)
bp = pup_mask[:, stims[1]]
el2 = cplt.compute_ellipse(pr2[bp, 0], pr2[bp, 1])
el_bp.plot(el2[0], el2[1], color='tab:orange', lw=lw)
el2 = cplt.compute_ellipse(pr2[~bp, 0], pr2[~bp, 1])
el_sp.plot(el2[0], el2[1], color='tab:orange', lw=lw)

el3 = cplt.compute_ellipse(pr3[:, 0], pr3[:, 1])
el_all.plot(el3[0], el3[1], color='tab:green', lw=lw)
el_all.scatter(pr3[:, 0], pr3[:, 1], color='tab:green', s=ms)
bp = pup_mask[:, stims[2]]
el3 = cplt.compute_ellipse(pr3[bp, 0], pr3[bp, 1])
el_bp.plot(el3[0], el3[1], color='tab:green', lw=lw)
el3 = cplt.compute_ellipse(pr3[~bp, 0], pr3[~bp, 1])
el_sp.plot(el3[0], el3[1], color='tab:green', lw=lw)


el_all.axhline(0, linestyle='--', color='k', zorder=-1)
el_all.axvline(0, linestyle='--', color='k', zorder=-1)

el_bp.axhline(0, linestyle='--', color='k', zorder=-1)
el_bp.axvline(0, linestyle='--', color='k', zorder=-1)

el_sp.axhline(0, linestyle='--', color='k', zorder=-1)
el_sp.axvline(0, linestyle='--', color='k', zorder=-1)

el_all.set_xlabel(r"Stim. $PC_1$")
el_all.set_ylabel(r"Stim. $PC_2$")
el_all.axis('square')
el_all.set_title("All Trials")

el_bp.set_xlabel(r"Stim. $PC_1$")
el_bp.axis('square')
el_bp.set_title("Large pupil trials")

el_sp.set_xlabel(r"Stim. $PC_1$")
el_sp.axis('square')
el_sp.set_title("Small pupil trials")

# force bp/sp to be the same size as all trials
el_bp.set_ylim(el_all.get_ylim())
el_bp.set_xlim(el_all.get_xlim())
el_sp.set_ylim(el_all.get_ylim())
el_sp.set_xlim(el_all.get_xlim())

f.tight_layout()

if savefig:
    f.savefig(fig_fn, dpi=400)

plt.show()