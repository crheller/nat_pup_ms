"""
experimental set up, example single trial population responses and pupil pupil trace.
    Also - gain modulation and delta noise correlation
    Consideration - show scatter plot of model performance, or should this be supplemental?
"""

import nems_lbhb.baphy as nb

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sf
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

site = 'TAR010c'
batch = 289
fs = 1000
options = {'batch': batch, 'cellid': site, 'pupil': True, 'stim': True, 'rasterfs': fs}
sigma = 30
rep1 = -12
rep2 = -2

rec = nb.baphy_load_recording_file(**options)
rec['resp'] = rec['resp'].rasterize()
rec['stim'] = rec['stim'].rasterize()
# extract ferretmixed41 epochs
r = rec['resp'].extract_epoch('STIM_00Oxford_male2b.wav')
spec = rec['stim'].extract_epoch('STIM_00Oxford_male2b.wav')
p = rec['pupil'].extract_epoch('STIM_00Oxford_male2b.wav')

# TODO - load waveform from baphy, generate spectrogram with gtgram function in nems

psth = sf.gaussian_filter1d(r.mean(axis=(0, 1)), sigma) * fs
psth1 = sf.gaussian_filter1d(r[rep1, :, :].mean(axis=0), sigma) * fs
psth2 = sf.gaussian_filter1d(r[rep2, :, :].mean(axis=0), sigma) * fs
spk_times1 = np.where(r[rep1, :, :])
spk_times2 = np.where(r[rep2, :, :])
mean_pupil1 = np.round(p[rep1].mean(axis=-1).squeeze(), 2)
mean_pupil2 = np.round(p[rep2].mean(axis=-1).squeeze(), 2)


# psths
f, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)

ax[0].plot(psth, color='grey', lw=2)
ax[0].plot(psth1, color='firebrick', lw=2)
ax[0].set_title(r"$\bar p_{k} = %s$" % mean_pupil1)

ax[1].plot(psth, color='grey', lw=2)
ax[1].plot(psth2, color='navy', lw=2)
ax[1].set_title(r"$\bar p_{k} = %s$" % mean_pupil2)

f.tight_layout()

# rasters
f, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)

ax[0].plot(spk_times1[1], spk_times1[0], '|', color='k', markersize=1)
ax[1].plot(spk_times2[1], spk_times2[0], '|', color='k', markersize=1)

f.tight_layout()

# spectrograms
f, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)

ax[0].imshow(spec[0, :, :], cmap='Greys', origin='lower', aspect='auto')
ax[1].imshow(spec[0, :, :], cmap='Greys', origin='lower', aspect='auto')

f.tight_layout()

# pupil distribution
plt.figure(figsize=(12, 3))
ax1 = plt.subplot2grid((3, 12), (0, 0), colspan=10, rowspan=3)
ax2 = plt.subplot2grid((3, 12), (0, 10), colspan=2, rowspan=3, sharey=ax1)

per_trial_pup = rec['pupil'].extract_epoch('REFERENCE').mean(axis=-1).squeeze()
bins = np.arange(0, 90, 5)
ax1.plot(rec['pupil']._data.T, color='forestgreen', lw=1)
ax2.hist(per_trial_pup, color='forestgreen', edgecolor='k', lw=1, rwidth=0.7, bins=bins, orientation='horizontal')

f.tight_layout()

plt.show()

