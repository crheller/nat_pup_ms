"""
experimental set up, example single trial population responses and pupil pupil trace.
    Also - gain modulation and delta noise correlation
    Consideration - show scatter plot of model performance, or should this be supplemental?
"""
from path_settings import PY_FIGURES_DIR2, PY_FIGURES_DIR3
from global_settings import ALL_SITES, HIGHR_SITES, CPN_SITES
import colors
from charlieTools.statistics import get_bootstrapped_sample, get_direct_prob
import load_results as ld

import nems_lbhb.baphy as nb
from nems_lbhb.plots import plot_weights_64D
from nems_lbhb.baphy_experiment import BAPHYExperiment

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sf
from scipy.io import wavfile
import nems.analysis.gammatone.gtgram as gt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8
#mpl.rcParams.update({'svg.fonttype': 'none'})

sites = HIGHR_SITES + CPN_SITES

savefig = True
fig_fn = PY_FIGURES_DIR3 + 'fig1_example.svg'
mi_bins = np.arange(-0.6, 0.6, 0.05)

f = plt.figure(figsize=(7.5, 3.5))
pax = plt.subplot2grid((6, 9), (0, 1), colspan=6, rowspan=2)
p1ax = plt.subplot2grid((6, 9), (2, 1), colspan=3, rowspan=4)
p2ax = plt.subplot2grid((6, 9), (2, 4), colspan=3, rowspan=4)
arr = plt.subplot2grid((6, 9), (2, 0), colspan=1, rowspan=4)
miax = plt.subplot2grid((6, 9), (0, 7), colspan=2, rowspan=3)
ncax = plt.subplot2grid((6, 9), (3, 7), colspan=2, rowspan=3)

site = 'TAR010c'
batch = 289
rasterfs = 1000
sigma = 30
rep1 = -12
rep2 = -2

manager = BAPHYExperiment(batch=batch, cellid=site)
rec = manager.get_recording(**{'rasterfs': rasterfs, 'pupil': True, 'resp': True})
rec['resp'] = rec['resp'].rasterize()
# extract epochs
soundfile = '/auto/users/hellerc/code/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set4/00cat668_rec7_ferret_oxford_male_chopped_excerpt1.wav'
epoch = 'STIM_00Oxford_male2b.wav'
r = rec['resp'].extract_epoch(epoch)
p = rec['pupil'].extract_epoch(epoch)

fs = rasterfs
psth = sf.gaussian_filter1d(r.mean(axis=(0, 1)), sigma) * fs
psth1 = sf.gaussian_filter1d(r[rep1, :, :].mean(axis=0), sigma) * fs
psth2 = sf.gaussian_filter1d(r[rep2, :, :].mean(axis=0), sigma) * fs
spk_times1 = np.where(r[rep1, :, :])
spk_times2 = np.where(r[rep2, :, :])
mean_pupil1 = p[rep1].mean(axis=-1).squeeze()
mean_pupil2 = p[rep2].mean(axis=-1).squeeze()

# psths
time = np.linspace(-2, (psth.shape[0] / rasterfs) - 2, psth.shape[0])
p1ax.plot(time, psth, color='grey', lw=1)
p1ax.plot(time, psth1, color='firebrick', lw=1)

p2ax.plot(time, psth, color='grey', lw=1)
p2ax.plot(time, psth2, color='navy', lw=1)


# plot units on array
plot_weights_64D(np.ones(len(rec['resp'].chans)), rec['resp'].chans,
                 overlap_method='mean', cbar=False, ax=arr, s=5)

# rasters
lim = 40
p1ax.plot((spk_times1[1] / rasterfs) - 2, lim + (spk_times1[0] / 1), '|', color='k', markersize=0.75, alpha=0.5, rasterized=True)
p2ax.plot((spk_times2[1] / rasterfs) - 2, lim + (spk_times2[0] / 1), '|', color='k', markersize=0.75, alpha=0.5, rasterized=True)

spec_offset = lim + ((spk_times1[0] / 1)).max() + 2
# spectrogram
fs, data = wavfile.read(soundfile)
# pad / crop data
data = data[:int(3 * fs)]
spbins = int(2 * fs)
postbins = int(0.5 * fs)
data = np.concatenate((np.zeros(spbins), data, np.zeros(postbins)))
spec = gt.gtgram(data, fs, 0.01, 0.002, 100, 0)
p1ax.imshow(np.sqrt(spec), cmap='Greys', origin='lower', aspect='auto', extent=[-2, 3.5, spec_offset, spec_offset+20])
p1ax.axvline(0, linestyle='--', lw=1, color='lime')
p1ax.axvline(3, linestyle='--', lw=1, color='lime')
p1ax.set_ylabel("Spk / s")
p1ax.set_title(r"$\bar p_{k} = %s$" % np.round(mean_pupil1, 2))
p1ax.set_xlabel('Time (s)')
p1ax.set_xlim((-2, 3.5))
p1ax.set_ylim((0, spec_offset+20))

p2ax.imshow(np.sqrt(spec), cmap='Greys', origin='lower', aspect='auto', extent=[-2, 3.5, spec_offset, spec_offset+20])
p2ax.axvline(0, linestyle='--', lw=1, color='lime')
p2ax.axvline(3, linestyle='--', lw=1, color='lime')
p2ax.set_ylabel("Spk / s")
p2ax.set_title(r"$\bar p_{k} = %s$" % np.round(mean_pupil2, 2))
p2ax.set_xlabel('Time (s)')
p2ax.set_xlim((-2, 3.5))
p2ax.set_ylim((0, spec_offset+20))

time = np.linspace(0, rec['pupil'].shape[-1] / rasterfs, rec['pupil'].shape[-1])
pdata = rec['pupil']._data.T
pax.plot(time, pdata, color='k', lw=0.5)
# add text to figure indicating when different stimulus excerpts played
epochs = [e for e in rec.epochs.name.unique() if 'STIM_00' in e]
yl = pax.get_ylim()[-1]
for l, e in zip(['a', 'b', 'c'], epochs):
    start = rec['pupil'].get_epoch_indices(e)[:, 0] / rasterfs
    for s in start:
        if e==epoch:
            col = 'lime'
        else:
            col = 'k'
        pax.text(s, yl, l, fontsize=4, color=col)
pax.spines['bottom'].set_visible(False)
pax.spines['left'].set_visible(False)
pax.set_xticks([])
pax.set_yticks([])
pax.set_ylim((pdata.min(), pdata.max()))

# plot MI
path = '/auto/users/hellerc/results/nat_pupil_ms/first_order_model_results/'
df = pd.concat([pd.read_csv(os.path.join(path,'d_289_pup_sdexp.csv'), index_col=0),
                pd.read_csv(os.path.join(path,'d_294_pup_sdexp.csv'), index_col=0),
                pd.read_csv(os.path.join(path,'d_331_pup_sdexp.csv'), index_col=0)])
try:
    df['r'] = [np.float(r.strip('[]')) for r in df['r'].values]
    df['r_se'] = [np.float(r.strip('[]')) for r in df['r_se'].values]
except:
    pass

df = df[df.state_chan=='pupil'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se'])
df['site'] = [idx[:7] for idx in df.index]
df = df[df.loc[:, 'site'].isin([s[:7] for s in sites])]
MI = df.loc[:, pd.IndexSlice['MI', 'st.pup']]
MI = np.array([np.float(x.strip('[]')) if type(x)==str else x for x in MI.values])

print(f"Mean MI: {MI.mean()}, sem: {MI.std() / np.sqrt(MI.shape[0])} \n n total units: {MI.shape[0]}")

miax.hist(MI, bins=mi_bins, edgecolor='white', color='grey')
miax.axvline(0, linestyle='--', color='k')
miax.set_xlabel('Modulation Index')
miax.set_ylabel(r"$n$ units")

# get bootstrapped p-value
np.random.normal(123)
d = {s: MI[np.argwhere(df['site'].values==s).squeeze()] for s in df['site'].unique()}
bootsample = get_bootstrapped_sample(d, even_sample=False, nboot=1000)
p = get_direct_prob(bootsample, np.zeros(len(bootsample)))[0]
miax.text(mi_bins.min(), miax.get_ylim()[-1]-2, r"p=%s" % round(p, 4))

# plot noise correlation
rsc_path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations_final/'
rsc_df = ld.load_noise_correlation('rsc_ev_perstim_mvm-25-1') #, xforms_model='NULL', path=rsc_path)
rsc_df = rsc_df[rsc_df.site.isin(CPN_SITES)]
rsc_df = pd.concat([rsc_df, ld.load_noise_correlation('rsc_ev_perstim')])
rsc_df = rsc_df[rsc_df.site.isin(sites)]
#rsc_df = rsc_df[(rsc_df.gm_bp>1) & (rsc_df.gm_sp>1)]
mask = ~(rsc_df['bp'].isna() | rsc_df['sp'].isna()) & (rsc_df['gm_bp']>1) & (rsc_df['gm_sp']>1)
rsc_df = rsc_df[mask]
d = {s: rsc_df.loc[rsc_df.site==s]['sp'].values - rsc_df.loc[rsc_df.site==s]['bp'].values for s in rsc_df.site.unique()}
bootsample = get_bootstrapped_sample(d, even_sample=False, nboot=1000)
p = get_direct_prob(bootsample, np.zeros(len(bootsample)))[0]

ncax.bar([0, 1], [rsc_df.groupby(by='site').mean()['bp'].mean(), rsc_df.groupby(by='site').mean()['sp'].mean()], 
                yerr=[rsc_df.groupby(by='site').mean()['bp'].sem(), rsc_df.groupby(by='site').mean()['sp'].sem()],
                color='none', edgecolor=[colors.LARGE, colors.SMALL], width=0.5)
ncax.set_ylabel('Noise correlation')
for s in sites:
    ncax.plot([0, 1], [rsc_df[rsc_df.site==s]['bp'].mean(), rsc_df[rsc_df.site==s]['sp'].mean()], 'grey')
ncax.text(ncax.get_xlim()[0], ncax.get_ylim()[-1], r"p=%s" % round(p, 4))

print(f"Mean rsc small: {rsc_df.groupby(by='site').mean()['sp'].mean()}, sem: {rsc_df.groupby(by='site').mean()['sp'].std() / np.sqrt(len(sites))}")
print(f"Mean rsc large: {rsc_df.groupby(by='site').mean()['bp'].mean()}, sem: {rsc_df.groupby(by='site').mean()['bp'].std() / np.sqrt(len(sites))}")
print(f"Mean unit pairs per session: {np.mean([len(x) for k, x in d.items()])}, sem: {np.std([len(x) for k, x in d.items()]) / len(d.keys())} ")

f.tight_layout()

if savefig:
    f.savefig(fig_fn, dpi=1000)

plt.show()

