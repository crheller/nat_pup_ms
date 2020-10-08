"""
Cartoon schematic of stimulus paradigm (ISI etc.) and procedure for chunking sounds and measuring dprime
Top row - spectrogram
Second row - raster plot
Third row - TDR space

Two columns, to allow for "..." in between
"""
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH
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

savefig = True
fig_fn1 = PY_FIGURES_DIR + 'fig3_stimulus_decoding_schemtic1.svg'
fig_fn2 = PY_FIGURES_DIR + 'fig3_stimulus_decoding_schemtic2.svg'

siteid = 'DRX006b.e1:64'
batch = 289
rasterfs = 1000
options = {'batch': batch, 'cellid': siteid, 'pupil': True, 'stim': False, 'rasterfs': rasterfs}
length = int(16 * 1.5)  # X stims * 1.5 sec (trial len)
twin1 = (int(rasterfs * 0), int(rasterfs * length))                   # first chunk of data
twin2 = (int(rasterfs * 5 * length), int(rasterfs * 6 * length))      # a later chunk of data
prestim = 0.25
poststim = 0.25
duration = 1
soundpath = '/auto/users/hellerc/code/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set4/'

rec = nb.baphy_load_recording_file(**options)
rec = preproc.mask_high_repetion_stims(rec)
rec = rec.apply_mask(reset_epochs=True)
rec['resp'] = rec['resp'].rasterize()
rec['resp'] = rec['resp'].extract_channels(rec.meta['cells_to_extract'])

early_epochs = rec.epochs[rec.epochs.name.str.contains('STIM_')]
early_epochs = early_epochs[(early_epochs.end <= (twin1[1] / rasterfs)) & \
                            (early_epochs.end >= (twin1[0] / rasterfs))]
late_epochs = rec.epochs[rec.epochs.name.str.contains('STIM_')]
late_epochs = late_epochs[(late_epochs.end <= (twin2[1] / rasterfs)) & \
                            (late_epochs.end >= (twin2[0] / rasterfs))]

ear_stims = early_epochs.name
t1 = early_epochs[['start', 'end']].values.tolist()
early_stimulus = []
spk_times_early = []
lat_stims = late_epochs.name
t2 = late_epochs[['start', 'end']].values.tolist()
late_stimulus = []
spk_times_late = []
S1 = list(set(ear_stims).intersection(set(lat_stims)))[0]
S2 = list(set(ear_stims).intersection(set(lat_stims)))[1]
sv_idx1 = []
sv_idx2 = []
sv_idx3 = []
sv_idx4 = []
for i, (epoch1, times1, epoch2, times2) in enumerate(zip(ear_stims, t1, lat_stims, t2)):
    print(f"loading sounds {i}/{len(ear_stims)}")
    soundfile = soundpath + epoch1.strip('STIM_')
    fs, data = wavfile.read(soundfile)
    # pad / crop data
    data = data[:int(duration * fs)]
    spbins = int(prestim * fs)
    postbins = int(poststim * fs)
    data = np.concatenate((np.zeros(spbins), data, np.zeros(postbins)))
    spec = gt.gtgram(data, fs, 0.01, 0.002, 100, 0)
    early_stimulus.append(spec)

    # get raster for this sound
    r = rec['resp'].extract_epoch(epoch1, mask=rec.and_mask((np.array([times1])*rasterfs).astype(int))['mask'])[0, :, :]
    n, st = np.where(r)
    st = st + int(i * (rasterfs * (duration + prestim + poststim)))
    spk_times_early.append((n, st))

    soundfile = soundpath + epoch2.strip('STIM_')
    fs, data = wavfile.read(soundfile)
    # pad / crop data
    data = data[:int(duration * fs)]
    spbins = int(prestim * fs)
    postbins = int(poststim * fs)
    data = np.concatenate((np.zeros(spbins), data, np.zeros(postbins)))
    spec = gt.gtgram(data, fs, 0.01, 0.002, 100, 0)
    late_stimulus.append(spec)

    # get raster for this sound
    r = rec['resp'].extract_epoch(epoch2, mask=rec.and_mask((np.array([times2])*rasterfs).astype(int))['mask'])[0, :, :]
    n, st = np.where(r)
    st = st + int(i * rasterfs * (duration + prestim + poststim))
    spk_times_late.append((n, st))

    if epoch1 == S1:
        sv_idx1.append(i)
    if epoch1 == S2:
        sv_idx2.append(i)
    if epoch2 == S1:
        sv_idx3.append(i)
    if epoch2 == S2:
        sv_idx4.append(i)

spec_slice_len = int((0.25 / 1.5) * early_stimulus[0].shape[-1]) # 250ms window to highlight
sss = spec_slice_len
sse = sss + spec_slice_len
ssl = int(4 * spec_slice_len)
chans = early_stimulus[0].shape[0]

highlight_early1 = [np.concatenate((np.nan * np.zeros((chans, sss)), e[:, sss:sse], np.nan * np.zeros((chans, ssl))), axis=-1) if i in sv_idx1 else np.nan * np.zeros(early_stimulus[0].shape) for i, e in enumerate(early_stimulus)]
highlight_early1 = np.concatenate(highlight_early1, axis=-1)
highlight_early2 = [np.concatenate((np.nan * np.zeros((chans, sss)), e[:, sss:sse], np.nan * np.zeros((chans, ssl))), axis=-1) if i in sv_idx2 else np.nan * np.zeros(early_stimulus[0].shape) for i, e in enumerate(early_stimulus)]
highlight_early2 = np.concatenate(highlight_early2, axis=-1)

highlight_late1 = [np.concatenate((np.nan * np.zeros((chans, sss)), e[:, sss:sse], np.nan * np.zeros((chans, ssl))), axis=-1) if i in sv_idx3 else np.nan * np.zeros(early_stimulus[0].shape) for i, e in enumerate(late_stimulus)]
highlight_late1 = np.concatenate(highlight_late1, axis=-1)
highlight_late2 = [np.concatenate((np.nan * np.zeros((chans, sss)), e[:, sss:sse], np.nan * np.zeros((chans, ssl))), axis=-1) if i in sv_idx4 else np.nan * np.zeros(early_stimulus[0].shape) for i, e in enumerate(late_stimulus)]
highlight_late2 = np.concatenate(highlight_late2, axis=-1)

early_stimulus = np.concatenate(early_stimulus, axis=-1)
late_stimulus = np.concatenate(late_stimulus, axis=-1)


rasterticksize = 0.75
marker = '|'
f = plt.figure(figsize=(10, 4))
spec1 = plt.subplot2grid((2, 2), (0, 0))
spec2 = plt.subplot2grid((2, 2), (0, 1))
ras1 = plt.subplot2grid((2, 2), (1, 0))
ras2 = plt.subplot2grid((2, 2), (1, 1))

spec1.imshow(np.sqrt(early_stimulus), cmap='Greys', origin='lower', aspect='auto')
spec1.imshow(np.sqrt(highlight_early1), cmap='Blues', origin='lower', aspect='auto')
spec1.imshow(np.sqrt(highlight_early2), cmap='Oranges', origin='lower', aspect='auto')
spec1.set_xticks([])
spec2.imshow(np.sqrt(late_stimulus), cmap='Greys', origin='lower', aspect='auto')
spec2.imshow(np.sqrt(highlight_late1), cmap='Blues', origin='lower', aspect='auto')
spec2.imshow(np.sqrt(highlight_late2), cmap='Oranges', origin='lower', aspect='auto')
spec2.set_xticks([])
spec2.set_yticks([])
spec2.spines['bottom'].set_visible(False)
spec2.spines['left'].set_visible(False)

for st in spk_times_early:
    ras1.plot((st[1] / rasterfs), (st[0] / 1), marker, color='k', markersize=rasterticksize, rasterized=True)
ras1.set_xlim((0, length))

for st in spk_times_late:
    ras2.plot((st[1] / rasterfs), (st[0] / 1), marker, color='k', markersize=rasterticksize, rasterized=True)
ras2.set_xlim((0, length))
ras2.set_yticks([])
ras2.set_xticks([])
ras2.spines['bottom'].set_visible(False)
ras2.spines['left'].set_visible(False)

ras1.set_xlabel("Time (s)")
ras1.set_ylabel("Neuron\n(sorted by depth)")
spec1.set_ylabel("Frequency")

f.tight_layout()

if savefig:
    f.savefig(fig_fn1)

# make a dumb schematic of the responses in TDR space, and plot the distributions on decoding axis
np.random.seed(123)
u1 = [-2, 0]
u2 = [2, 0]
cov = np.array([[1, 0.5], [0.5, 1]])

A = np.random.multivariate_normal(u1, cov, (20,))
B = np.random.multivariate_normal(u2, cov, (20,))
Ael = cplt.compute_ellipse(A[:, 0], A[:, 1])
Bel = cplt.compute_ellipse(B[:, 0], B[:, 1])

f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(A[:, 0], A[:, 1], edgecolor='white')
ax[0].plot(Ael[0], Ael[1], color='tab:blue', lw=2)
ax[0].scatter(B[:, 0], B[:, 1], edgecolor='white')
ax[0].plot(Bel[0], Bel[1], color='tab:orange', lw=2)

ax[0].axis('equal')

ax[0].set_xlabel(r"$\Delta \mathbf{\mu}$ ($TDR_1$)")
ax[0].set_ylabel(r"$TDR_2$")

_, wopt, _, _, _, _ = decoding.compute_dprime(A.T, B.T)

x = np.arange(-20, 20, 0.001)
u1 = A.dot(wopt).mean()
sd1 = A.dot(wopt).std()
u2 = B.dot(wopt).mean()
sd2 = B.dot(wopt).std()
ax[1].plot(x, np.exp(-.5 * ((x - u1) / sd1) ** 2), color='tab:blue', lw=2)
ax[1].plot(x, np.exp(-.5 * ((x - u2) / sd2) ** 2), color='tab:orange', lw=2)
ax[1].set_xlabel('Discrimination axis')
ax[1].set_ylim((0, 2))

if savefig:
    f.savefig(fig_fn2)

plt.show()