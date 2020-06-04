"""
experimental set up, example single trial population responses and pupil pupil trace.
    Also - gain modulation and delta noise correlation
    Consideration - show scatter plot of model performance, or should this be supplemental?
"""

import nems_lbhb.baphy as nb
from nems_lbhb.plots import plot_weights_64D

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sf
from scipy.io import wavfile
import nems.analysis.gammatone.gtgram as gt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams.update({'svg.fonttype': 'none'})

pupil_heatmap = False

savefig = False
fig_fn = '/home/charlie/Desktop/lbhb/code/projects/nat_pup_ms/py_figures/fig1_example.svg'

if pupil_heatmap:
    f = plt.figure(figsize=(8, 6))
    # leave first subplot blk (for experimental setup)
    pax = plt.subplot2grid((10, 4), (1, 3), rowspan=8)
    spax = plt.subplot2grid((10, 4), (0, 1), colspan=2, rowspan=2)
    p1ax = plt.subplot2grid((10, 4), (2, 1), colspan=2, rowspan=4)
    p2ax = plt.subplot2grid((10, 4), (6, 1), colspan=2, rowspan=4)
    arr = plt.subplot2grid((10, 4), (4, 0), rowspan=6)
else:
    f = plt.figure(figsize=(10, 6))
    pax = plt.subplot2grid((4, 10), (0, 2), colspan=8, rowspan=1)
    spax = plt.subplot2grid((4, 10), (1, 2), colspan=4)
    spax2 = plt.subplot2grid((4, 10), (1, 6), colspan=4)
    p1ax = plt.subplot2grid((4, 10), (2, 2), colspan=4, rowspan=2)
    p2ax = plt.subplot2grid((4, 10), (2, 6), colspan=4, rowspan=2)
    arr = plt.subplot2grid((4, 10), (1, 0), colspan=2, rowspan=3)

site = 'TAR010c'
batch = 289
rasterfs = 1000
options = {'batch': batch, 'cellid': site, 'pupil': True, 'stim': True, 'rasterfs': rasterfs}
sigma = 30
rep1 = -12
rep2 = -2

rec = nb.baphy_load_recording_file(**options)
rec['resp'] = rec['resp'].rasterize()
rec['stim'] = rec['stim'].rasterize()
# extract epochs
soundfile = '/auto/users/hellerc/code/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set4/00cat668_rec7_ferret_oxford_male_chopped_excerpt1.wav'
r = rec['resp'].extract_epoch('STIM_00Oxford_male2b.wav')
spec = rec['stim'].extract_epoch('STIM_00Oxford_male2b.wav')
p = rec['pupil'].extract_epoch('STIM_00Oxford_male2b.wav')

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
p1ax.plot(time, psth, color='grey', lw=2)
p1ax.plot(time, psth1, color='firebrick', lw=2)
p1ax.axvline(0, linestyle='--', lw=2, color='lime')
p1ax.axvline(3, linestyle='--', lw=2, color='lime')
p1ax.set_ylabel("Spk / s")
p1ax.set_title(r"$\bar p_{k} = %s$" % np.round(mean_pupil1, 2))
p1ax.set_xlabel('Time (s)')
p1ax.set_xlim((-2, 3.5))

p2ax.plot(time, psth, color='grey', lw=2)
p2ax.plot(time, psth2, color='navy', lw=2)
p2ax.axvline(0, linestyle='--', lw=2, color='lime')
p2ax.axvline(3, linestyle='--', lw=2, color='lime')
p2ax.set_ylabel("Spk / s")
p2ax.set_title(r"$\bar p_{k} = %s$" % np.round(mean_pupil2, 2))
p2ax.set_xlabel('Time (s)')
p2ax.set_xlim((-2, 3.5))

# plot units on array
plot_weights_64D(np.ones(len(rec['resp'].chans)), rec['resp'].chans,
                 overlap_method='mean', cbar=False, ax=arr, s=50)

# rasters
lim = 40
p1ax.plot((spk_times1[1] / rasterfs) - 2, lim + (spk_times1[0] / 1), '|', color='k', markersize=0.75)
p2ax.plot((spk_times2[1] / rasterfs) - 2, lim + (spk_times2[0] / 1), '|', color='k', markersize=0.75)

# spectrogram
fs, data = wavfile.read(soundfile)
# pad / crop data
data = data[:int(3 * fs)]
spbins = int(2 * fs)
postbins = int(0.5 * fs)
data = np.concatenate((np.zeros(spbins), data, np.zeros(postbins)))
spec = gt.gtgram(data, fs, 0.01, 0.002, 100, 0)
spax.imshow(np.sqrt(spec), cmap='Greys', origin='lower', aspect='auto')
spax.spines['bottom'].set_visible(False)
spax.spines['left'].set_visible(False)
spax.set_xticks([])
spax.set_yticks([])

if not pupil_heatmap:
    spax2.imshow(np.sqrt(spec), cmap='Greys', origin='lower', aspect='auto')
    spax2.spines['bottom'].set_visible(False)
    spax2.spines['left'].set_visible(False)
    spax2.set_xticks([])
    spax2.set_yticks([])

if pupil_heatmap:
    # plot pupil as heatmap
    p = rec['pupil'].extract_epoch('REFERENCE').squeeze()
    p1_idx = np.argwhere(p.mean(axis=-1).squeeze()==mean_pupil1)[0][0]
    p2_idx = np.argwhere(p.mean(axis=-1).squeeze()==mean_pupil2)[0][0]
    im = pax.imshow(p, aspect='auto', cmap='Purples', origin='lower')
    pax.axhline(p1_idx, lw=2, color='firebrick')
    pax.axhline(p2_idx, lw=2, color='navy')
    pax.set_xticks([0*rasterfs, 2*rasterfs, 5*rasterfs])
    pax.set_xticklabels([-2, 0, 3])
    pax.set_ylabel(r'Trial $k$')
    pax.set_xlabel('Time (s)')
    pax.set_title("Pupil Size")
    f.colorbar(im, ax=pax)

else:
    time = np.linspace(0, rec['pupil'].shape[-1] / rasterfs, rec['pupil'].shape[-1])
    pax.plot(time, rec['pupil']._data.T, color='k')
    pax.set_ylabel('Pupil Size')
    pax.set_xlabel('Time (s)')

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()

