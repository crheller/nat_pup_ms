import matplotlib.pyplot as plt
import scipy.signal as ss
import pandas as pd
import numpy as np

import nems.db as nd
import nems.xforms as xforms

import charlieTools.plotting as cplt

batch = 289
sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
        'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
        'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b', 'BOL005c', 'BOL006b']
F = []
Fm = []
S = []
Sm = []
P = []
Pm = []

for site in sites:

    a = 'af0:4.as0:4.sc.rb10'
    best_alpha = pd.read_csv('/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime/best_alpha.csv', index_col=0)
    alpha = best_alpha.loc[site][0]
    alpha = (float(alpha.split(',')[0].replace('(','')), float(alpha.split(',')[1].replace(')','')))
    a = 'af{0}.as{1}.sc.rb10'.format(str(alpha[0]).replace('.', ':'), str(alpha[1]).replace('.',':'))
    modelname = 'ns.fs4.pup-ld-hrc-apm-pbal-psthfr-ev-residual-addmeta_lv.2xR.f.s-lvlogsig.3xR.ipsth_jk.nf5.p-pupLVbasic.constrLVonly.{}'.format(a)

    cellid = [c for c in nd.get_batch_cells(batch).cellid if site in c][0]
    mp = nd.get_results_file(batch, [modelname], [cellid]).modelpath[0]

    xfspec, ctx = xforms.load_analysis(mp)

    r = ctx['val'].apply_mask()
    fs = r['resp'].fs
    fast = r['lv'].extract_channels(['lv_fast'])._data.squeeze()
    slow = r['lv'].extract_channels(['lv_slow'])._data.squeeze()
    pupil = r['pupil']._data.squeeze()


    o = ss.periodogram(fast, fs=fs)
    F.append(o[1].squeeze())
    Fm.append(o[0][np.argmax(o[1].squeeze())])

    o = ss.periodogram(slow, fs=fs)
    S.append(o[1].squeeze())
    Sm.append(o[0][np.argmax(o[1].squeeze())])

    o = ss.periodogram(pupil, fs=fs)
    P.append(o[1].squeeze())
    Pm.append(o[0][np.argmax(o[1].squeeze())])


# plot average
n = 80
freq = np.linspace(0, 2, n)
# down sample to min resolution
F = np.concatenate([ss.resample(f, n)[:, np.newaxis] for f in F], axis=-1)
S = np.concatenate([ss.resample(s, n)[:, np.newaxis] for s in S], axis=-1)
P = np.concatenate([ss.resample(p, n)[:, np.newaxis] for p in P], axis=-1)

F /= F.max(axis=0)
S /= S.max(axis=0)
P /= P.max(axis=0)

f, ax = plt.subplots(1, 3)

ax[0].plot(freq, F.mean(axis=-1), color='k')
sem = F.std(axis=-1) / np.sqrt(F.shape[-1])
ax[0].fill_between(freq, F.mean(axis=-1)-sem, F.mean(axis=-1)+sem,
                    color='lightgrey')
ax[0].scatter(Fm, np.ones(len(Fm)), edgecolor='white', color='k', s=15)
ax[0].set_title('Fast LV', fontsize=8)
ax[0].set_ylabel('Power', fontsize=8)
ax[0].set_xlabel('Freq (Hz)', fontsize=8)
ax[0].set_ylim((0, 1.05))
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(freq, S.mean(axis=-1), color='k')
sem = S.std(axis=-1) / np.sqrt(S.shape[-1])
ax[1].fill_between(freq, S.mean(axis=-1)-sem, S.mean(axis=-1)+sem,
                    color='lightgrey')
ax[1].scatter(Sm, np.ones(len(Sm)), edgecolor='white', color='k', s=15)
ax[1].set_title('Slow LV', fontsize=8)
ax[1].set_ylabel('Power', fontsize=8)
ax[1].set_xlabel('Freq (Hz)', fontsize=8)
ax[1].set_ylim((0, 1.05))
ax[1].set_aspect(cplt.get_square_asp(ax[1]))


ax[2].plot(freq, P.mean(axis=-1), color='k')
sem = F.std(axis=-1) / np.sqrt(P.shape[-1])
ax[2].fill_between(freq, P.mean(axis=-1)-sem, P.mean(axis=-1)+sem,
                    color='lightgrey')
ax[2].scatter(Pm, np.ones(len(Pm)), edgecolor='white', color='k', s=15)
ax[2].set_title('Pupil', fontsize=8)
ax[2].set_ylabel('Power', fontsize=8)
ax[2].set_xlabel('Freq (Hz)', fontsize=8)
ax[2].set_ylim((0, 1.05))
ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()

plt.show()