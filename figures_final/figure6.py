"""
Compare first order pupil modulation to second order modulation.
Overlapping populations of neurons? Or mostly independent.

To do this, define first order axis as first PC of model residual.
Plot projection on this axis to give intuition?
"""
from global_settings import CPN_SITES
from path_settings import PY_FIGURES_DIR3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

from nems.xform_helper import load_model_xform
import nems.db as nd

savefig = True
fig_fn = PY_FIGURES_DIR3 + 'fig6.svg'

batch = 331
results = {
    'pca': {},
    'lvax': {},
    'lv0ax': {},
    'lvbax': {}
}
for site in CPN_SITES:
    model = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
    xf, ctx = load_model_xform(cellid=site, modelname=model, batch=331)

    # first order prediction -- what's the dimensionality of the residual?
    r0 = ctx['val']
    r0 = r0.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)

    res0 = r0.apply_mask()['pred0']._data - r0.apply_mask()['psth_sp']._data 

    # do PCA and plot scree plot to check dimensionality
    pca = PCA()
    pca.fit(res0.T)

    # get LV axis
    lvax = ctx['modelspec'][0]['phi']['g'][:, 1]
    lvax /= np.linalg.norm(lvax)  # get unit vector for comparison with pcs

    # Look at shuffled pup dimension as controal
    lv0ax = ctx['modelspec'][0]['phi']['g'][:, 2]
    lv0ax /= np.linalg.norm(lv0ax)

    # get baseline axis
    lvbax = ctx['modelspec'][0]['phi']['g'][:, 0]
    lvbax /= lvbax

    results['pca'][site] = pca
    results['lvax'][site] = lvax
    results['lv0ax'][site] = lv0ax
    results['lvbax'][site] = lvbax


# MAKE FIGURE
cmap = {
    'lv': 'indigo',
    'lv0': 'goldenrod'
}
ms = 10
alpha = 0.3
edgecolor = 'none'
f, ax = plt.subplots(1, 3, figsize=(6.2, 2.5))

# plot scree plot over pupil modulation of sensory responses
for s in CPN_SITES:
    ax[0].plot(np.cumsum(results['pca'][s].explained_variance_ratio_), alpha=0.5, color='tab:blue')
ax[0].set_ylim((0, 1.1))
ax[0].set_ylabel('Cum. fraction of sensory\nresponse modulation exp.')
ax[0].set_xlabel("Principal Component")

# plot model weights
x = []
ylv = []
ylv0 = []
for s in CPN_SITES:
    pc0 = results['pca'][s].components_[0] * np.sign(np.mean(results['pca'][s].components_[0]))
    ax[1].scatter(pc0, results['lvax'][s] * np.sign(np.mean(results['lvax'][s])), s=ms, color=cmap['lv'], alpha=alpha, edgecolor='none')
    ax[2].scatter(pc0, results['lv0ax'][s] * np.sign(np.mean(results['lv0ax'][s])), s=ms, color=cmap['lv0'], alpha=alpha, edgecolor='none')
    x.append(pc0)
    ylv.append(results['lvax'][s] * np.sign(np.mean(results['lvax'][s])))
    ylv0.append(results['lv0ax'][s] * np.sign(np.mean(results['lv0ax'][s])))
x = np.concatenate(x); ylv = np.concatenate(ylv); ylv0 = np.concatenate(ylv0)
ax[1].axhline(0, linestyle='--', color='grey', zorder=-1); ax[1].axvline(0, linestyle='--', color='grey', zorder=-1)
ax[2].axhline(0, linestyle='--', color='grey', zorder=-1); ax[2].axvline(0, linestyle='--', color='grey', zorder=-1)

ax[1].text(-0.5, 1, r"r=%s"%round(np.corrcoef(x, ylv)[0, 1], 3))
ax[2].text(-0.5, 1, r"r=%s"%round(np.corrcoef(x, ylv0)[0, 1], 3))

ax[1].set_xlabel(r"$PC_1$")
ax[1].set_ylabel(r"Shared modulator weight")
ax[1].set_title("Pupil-dependent\nmodulator")
ax[2].set_xlabel(r"$PC_1$")
ax[2].set_ylabel(r"Shared modulator weight")
ax[2].set_title("Pupil-independent\nmodulator")

mm = np.max((ax[1].get_xlim()+ax[2].get_xlim()+ax[1].get_ylim()+ax[2].get_ylim()))
mi = np.min((ax[1].get_xlim()+ax[2].get_xlim()+ax[1].get_ylim()+ax[2].get_ylim()))
ax[1].set_xlim((mi, mm)); ax[1].set_ylim((mi, mm))
ax[2].set_xlim((mi, mm)); ax[2].set_ylim((mi, mm))

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

plt.show()