"""
1 - Spectrogram / response raster for single trial data
2 - Stim PC1/2 ellipse plots: overall, big, and small pupil
    - only show single trials for the highlighted chunks in the
      spectrogram
3 - scatter plot of big / small pupil dprime
? - add schematic of dprime calculation onto the ellipse plot?
"""

from path_settings import DPRIME_DIR, PY_FIGURES_DIR2, CACHE_PATH
import charlieTools.plotting as cplt
import charlieTools.decoding as decoding
import nems_lbhb.baphy as nb
from nems_lbhb.plots import plot_weights_64D
import nems_lbhb.preprocessing as preproc
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sf
from scipy.io import wavfile
import nems.analysis.gammatone.gtgram as gt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

e1 = ('STIM_00ferretmixed41.wav', 9)
e2 = ('STIM_00ferretmixed41.wav', 12)
e3 = ('STIM_00Oxford_male2b.wav', 19)

siteid = 'TAR010c'
batch = 289
rasterfs = 1000
options = {'batch': batch, 'cellid': siteid, 'pupil': True, 'stim': False, 'rasterfs': rasterfs}
length = 6 * 5.5  # X stims * 5.5 sec (trial len)
start = int(rasterfs * 160)
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
rec['resp'] = rec['resp'].rasterize()
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


# Make figure
f = plt.figure(figsize=(7, 6))

spec = plt.subplot2grid((3, 4), (0, 0), colspan=4)
rast = plt.subplot2grid((3, 4), (1, 0), colspan=4)

spec.imshow(np.sqrt(np.concatenate(stimulus, axis=-1)), 
                        origin='lower', cmap='Greys', aspect='auto')
spec.imshow(np.sqrt(np.concatenate(stim1, axis=-1)), 
                        origin='lower', cmap='Blues', aspect='auto')
spec.imshow(np.sqrt(np.concatenate(stim2, axis=-1)), 
                        origin='lower', cmap='Oranges', aspect='auto')
spec.imshow(np.sqrt(np.concatenate(stim3, axis=-1)), 
                        origin='lower', cmap='Greens', aspect='auto')
spec.set_xticks([])

ms = 1
rast.plot(np.concatenate(spk_times, axis=-1)[1, :] / rasterfs, 
            np.concatenate(spk_times, axis=-1)[0, :], 
            '|', color='k', markersize=ms)
rast.plot(np.concatenate(r1, axis=-1)[1, :] / rasterfs, 
            np.concatenate(r1, axis=-1)[0, :], 
            '|', color='tab:blue', markersize=ms)
rast.plot(np.concatenate(r2, axis=-1)[1, :] / rasterfs, 
            np.concatenate(r2, axis=-1)[0, :], 
            '|', color='tab:orange', markersize=ms)
rast.plot(np.concatenate(r3, axis=-1)[1, :] / rasterfs, 
            np.concatenate(r3, axis=-1)[0, :], 
            '|', color='tab:green', markersize=ms)
rast.set_xlim((0, length))

f.tight_layout()

plt.show()