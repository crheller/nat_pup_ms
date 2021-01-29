"""
1 - Spectrogram / response raster for single trial data
2 - Stim PC1/2 ellipse plots: overall, big, and small pupil
    - only show single trials for the highlighted chunks in the
      spectrogram
3 - scatter plot of big / small pupil dprime
? - add schematic of dprime calculation onto the ellipse plot?
"""
from global_settings import HIGHR_SITES, LOWR_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR2, CACHE_PATH
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
from scipy.stats import gaussian_kde
import os
import copy
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

e1 = ('STIM_00ferretmixed41.wav', 8)
e2 = ('STIM_00ferretmixed41.wav', 12)
e3 = ('STIM_00Oxford_male2b.wav', 12)

siteid = 'TAR010c'
batch = 289
rasterfs = 1000
options = {'batch': batch, 'cellid': siteid, 'pupil': True, 'stim': False, 'rasterfs': rasterfs}
length = 4 * 5.5  # X stims * 5.5 sec (trial len)
start = int(rasterfs * 240)
end = start + int(length * rasterfs)
twin = (start, end)      # a chunk of data
prestim = 2 #0.25
poststim = 0.5 #0.25
duration = 3
#soundpath = '/auto/users/hellerc/code/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set4/'
soundpath = '/auto/users/hellerc/code/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds/'

rec = nb.baphy_load_recording_file(**options)
rec = preproc.mask_high_repetion_stims(rec)
rec = rec.apply_mask(reset_epochs=True)
rec['resp'] = rec['resp'].extract_channels(rec.meta['cells_to_extract'])

epochs = rec.epochs[rec.epochs.name.str.contains('STIM_')]
epochs = epochs[(epochs.end <= (twin[1] / rasterfs)) & \
                            (epochs.end >= (twin[0] / rasterfs))]


stims = epochs.name
t = epochs[['start', 'end']].values.tolist()
stimulus = []
spk_times = []
# for saving the highlighted data
stim1 = []
stim2 = []
stim3 = []
r1 = []
r2 = []
r3 = []
offset = 0
for i, (epoch, times) in enumerate(zip(stims, t)):
    print(f"Loading data for epoch number {i}, name: {epoch}")
    # get spectrogram for this epoch
    soundfile = soundpath + epoch.strip('STIM_')
    fs, data = wavfile.read(soundfile)
    # pad / crop data
    data = data[:int(duration * fs)]
    spbins = int(prestim * fs)
    postbins = int(poststim * fs)
    data = np.concatenate((np.zeros(spbins), data, np.zeros(postbins)))
    spec = gt.gtgram(data, fs, 0.01, 0.002, 100, 0)
    stimulus.append(spec)

    # get raster for this sound
    r = rec['resp'].extract_epoch(epoch, mask=rec.and_mask((np.array([times])*rasterfs).astype(int))['mask'])[0, :, :]
    n, st = np.where(r)
    offset = int(i * (rasterfs * (duration + prestim + poststim)))
    st = st + offset
    #offset = st.max()
    spk_times.append((n, st))

    # get highlighted data
    s1 = np.zeros(spec.shape) * np.nan
    s2 = np.zeros(spec.shape) * np.nan
    s3 = np.zeros(spec.shape) * np.nan
    if epoch == e1[0]:
        nbins = 4 * (prestim + poststim + duration)
        bins = np.linspace(0, nbins, spec.shape[-1])
        sidx = np.argmin(np.abs(e1[1]-bins))
        eidx = np.argmin(np.abs((e1[1]+1)-bins))
        s1[:, sidx:eidx] = spec[:, sidx:eidx]
        # spikes
        ts = int((e1[1] / 4) * rasterfs) + offset
        te = int(((e1[1] / 4) + 0.25) * rasterfs) + offset
        tspks = np.argwhere((st<ts) | (st>te)).squeeze()
        spk_times[-1] = (n[tspks], st[tspks])
        tspks = np.argwhere((st>=ts) & (st<=te)).squeeze()
        r1.append((n[tspks], st[tspks]))
    if epoch == e2[0]:
        nbins = 4 * (prestim + poststim + duration)
        bins = np.linspace(0, nbins, spec.shape[-1])
        sidx = np.argmin(np.abs(e2[1]-bins))
        eidx = np.argmin(np.abs((e2[1]+1)-bins))
        s2[:, sidx:eidx] = spec[:, sidx:eidx]
        # spikes
        ts = int((e2[1] / 4) * rasterfs) + offset
        te = int(((e2[1] / 4) + 0.25) * rasterfs) + offset
        tspks = np.argwhere((st<ts) | (st>te)).squeeze()
        spk_times[-1] = (n[tspks], st[tspks])
        tspks = np.argwhere((st>=ts) & (st<=te)).squeeze()
        r2.append((n[tspks], st[tspks]))
    if epoch == e3[0]:
        nbins = 4 * (prestim + poststim + duration)
        bins = np.linspace(0, nbins, spec.shape[-1])
        sidx = np.argmin(np.abs(e3[1]-bins))
        eidx = np.argmin(np.abs((e3[1]+1)-bins))
        s3[:, sidx:eidx] = spec[:, sidx:eidx]
        # spikes
        ts = int((e3[1] / 4) * rasterfs) + offset
        te = int(((e3[1] / 4) + 0.25) * rasterfs) + offset
        tspks = np.argwhere((st<ts) | (st>te)).squeeze()
        spk_times[-1] = (n[tspks], st[tspks])
        tspks = np.argwhere((st>=ts) & (st<=te)).squeeze()
        r3.append((n[tspks], st[tspks]))
        
    stim1.append(s1)
    stim2.append(s2)
    stim3.append(s3)

# ====================== Get PC data ======================
X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=siteid, 
                                                         batch=batch, 
                                                         return_epoch_list=True)
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
pup_mask = pup_mask.reshape(1, nreps, nstim).squeeze()
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
proj = (X_spont).T.dot(pca.components_[0:2, :].T)

pr1 = proj[np.argwhere(np.array(epochs)==e1[0])[0][0] * nbins + e1[1]]
pr2 = proj[np.argwhere(np.array(epochs)==e2[0])[0][0] * nbins + e2[1]]
pr3 = proj[np.argwhere(np.array(epochs)==e3[0])[0][0] * nbins + e3[1]]

# figure out tseries projection to go under raster
ts = []
for time, stim in zip(t, stims):
    tt = rec['resp'].extract_epoch(stim, 
                    mask=rec.and_mask((np.array([time])*rasterfs).astype(int))['mask'])
    bins = int(tt.shape[-1] / (rasterfs / 4)) - 1
    print(bins)
    tt = np.apply_along_axis(lambda a: np.histogram(np.where(a), bins=bins)[0], -1, tt)
    ts.append(tt)
ts = (np.concatenate(ts, axis=0).transpose([1, 0, 2]).reshape(ncells, -1).T - spont.squeeze()).T
tp1 = ts.T.dot(pca.components_[0, :])
tp2 = ts.T.dot(pca.components_[1, :])

#(divide by arbitrary scaling factor)
tp1 /= 2
tp2 /= 2

# ============================= LOAD DPRIME =========================================
path = DPRIME_DIR
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'
n_components = 2
recache = False
df = []
for site in HIGHR_SITES:
    if (site in LOWR_SITES):
        mn = modelname.replace('_jk10', '_jk1_eev')
    else:
        mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df = results.numeric_results

    stim = results.evoked_stimulus_pairs
    _df = _df.loc[pd.IndexSlice[stim, 2], :]
    _df['site'] = site
    df.append(_df)

df = pd.concat(df)


# ============================== Make figure =========================================
f = plt.figure(figsize=(7, 5))

rast = plt.subplot2grid((5, 4), (0, 0), rowspan=3, colspan=4)
el_all = plt.subplot2grid((5, 4), (3, 0), rowspan=2, colspan=1)
el_bp = plt.subplot2grid((5, 4), (3, 1), rowspan=2, colspan=1)
el_sp = plt.subplot2grid((5, 4), (3, 2), rowspan=2, colspan=1)
scax = plt.subplot2grid((5, 4), (3, 3), rowspan=2, colspan=1)

# OFFSET FOR PC TIMESERIES PLOT
offset = np.max(np.concatenate([tp1, tp2])) + 1

# plot spectrogram
ext = [0, length, ncells+offset, ncells+int(stimulus[0].shape[0] / 4)+offset]
rast.imshow(np.sqrt(np.concatenate(stimulus, axis=-1)), 
                        origin='lower', cmap='Greys', aspect='auto', extent=ext)
rast.imshow(np.sqrt(np.concatenate(stim1, axis=-1)), 
                        origin='lower', cmap='Blues', aspect='auto', extent=ext)
rast.imshow(np.sqrt(np.concatenate(stim2, axis=-1)), 
                        origin='lower', cmap='Oranges', aspect='auto', extent=ext)
rast.imshow(np.sqrt(np.concatenate(stim3, axis=-1)), 
                        origin='lower', cmap='Greens', aspect='auto', extent=ext)

# plot raster ticks
argsort = np.argsort(np.abs(pca.components_[0]))
argmap = {argsort[i]: i for i in range(ncells)}
mapper = lambda x: argmap[x]
mfunc = np.vectorize(mapper)
ms = 0.8
m = '|'
rast.plot(np.concatenate(spk_times, axis=-1)[1, :] / rasterfs, 
            mfunc(np.concatenate(spk_times, axis=-1)[0, :]) + offset, 
            m, color='k', markersize=ms, alpha=0.4)
rast.plot(np.concatenate(r1, axis=-1)[1, :] / rasterfs, 
            mfunc(np.concatenate(r1, axis=-1)[0, :]) + offset, 
            m, color='tab:blue', markersize=ms, alpha=0.4)
rast.plot(np.concatenate(r2, axis=-1)[1, :] / rasterfs, 
            mfunc(np.concatenate(r2, axis=-1)[0, :]) + offset, 
            m, color='tab:orange', markersize=ms, alpha=0.4)
rast.plot(np.concatenate(r3, axis=-1)[1, :] / rasterfs, 
            mfunc(np.concatenate(r3, axis=-1)[0, :]) + offset, 
            m, color='tab:green', markersize=ms, alpha=0.4)
rast.set_xlim((0, length))

# plot pc timeseries
rast.plot(np.linspace(0, length, tp1.shape[0]), tp1, color='k', lw=2, label=r'Stim. $PC_1$')
rast.plot(np.linspace(0, length, tp1.shape[0]), tp2, color='grey', lw=2, label=r'Stim. $PC_2$')
rast.axhline(0, linestyle='--', lw=0.8, color='k', label='Spont baseline')

rast.legend(frameon=False, bbox_to_anchor=(1, 0), loc='lower left',)

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
bp = pup_mask[:, np.argwhere(np.array(epochs)==e1[0])[0][0] * nbins + e1[1]]
el1 = cplt.compute_ellipse(pr1[bp, 0], pr1[bp, 1])
el_bp.plot(el1[0], el1[1], color='tab:blue', lw=lw)
el1 = cplt.compute_ellipse(pr1[~bp, 0], pr1[~bp, 1])
el_sp.plot(el1[0], el1[1], color='tab:blue', lw=lw)

el2 = cplt.compute_ellipse(pr2[:, 0], pr2[:, 1])
el_all.plot(el2[0], el2[1], color='tab:orange', lw=lw)
el_all.scatter(pr2[:, 0], pr2[:, 1], color='tab:orange', s=ms)
bp = pup_mask[:, np.argwhere(np.array(epochs)==e2[0])[0][0] * nbins + e2[1]]
el2 = cplt.compute_ellipse(pr2[bp, 0], pr2[bp, 1])
el_bp.plot(el2[0], el2[1], color='tab:orange', lw=lw)
el2 = cplt.compute_ellipse(pr2[~bp, 0], pr2[~bp, 1])
el_sp.plot(el2[0], el2[1], color='tab:orange', lw=lw)

el3 = cplt.compute_ellipse(pr3[:, 0], pr3[:, 1])
el_all.plot(el3[0], el3[1], color='tab:green', lw=lw)
el_all.scatter(pr3[:, 0], pr3[:, 1], color='tab:green', s=ms)
bp = pup_mask[:, np.argwhere(np.array(epochs)==e3[0])[0][0] * nbins + e3[1]]
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
el_bp.set_ylabel(r"Stim. $PC_2$")
el_bp.axis('square')
el_bp.set_title("Large pupil trials")

el_sp.set_xlabel(r"Stim. $PC_1$")
el_sp.set_ylabel(r"Stim. $PC_2$")
el_sp.axis('square')
el_sp.set_title("Small pupil trials")

# force bp/sp to be the same size as all trials
el_bp.set_ylim(el_all.get_ylim())
el_bp.set_xlim(el_all.get_xlim())
el_sp.set_ylim(el_all.get_ylim())
el_sp.set_xlim(el_all.get_xlim())

# plot dprime results
nSamples = 2000
idx = df[['bp_dp', 'sp_dp']].max(axis=1) < 100
sidx = np.random.choice(range(idx.sum()), nSamples, replace=False)
bp = df['bp_dp'].values[idx][sidx]
sp = df['sp_dp'].values[idx][sidx]
s = 5
xy = np.vstack([bp, sp])
z = gaussian_kde(xy)(xy)
scax.scatter(sp, bp, s=s, c=z)
scax.plot([0, 100], [0, 100], 'k--')
scax.set_xlabel("Small pupil")
scax.set_ylabel("Large pupil")
scax.set_title(r"Stimulus discriminability ($d'^2$)")
scax.axis('square')

f.tight_layout()

plt.show()