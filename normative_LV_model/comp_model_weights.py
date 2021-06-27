"""
Attempt to compare first and second order model weights.
"""
from global_settings import CPN_SITES
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from nems.xform_helper import load_model_xform
import nems.db as nd

batch = 331
for site in CPN_SITES:
    model0 = 'psth.fs4.pup-ld-st.pup-epcpn-mvm.25.2-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000'
    model = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
    model00 = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"

    #xf0, ctx0 = load_model_xform(cellid=[c for c in nd.get_batch_cells(batch).cellid if site in c][0], modelname=model0, batch=331)
    xf, ctx = load_model_xform(cellid=site, modelname=model, batch=331)
    xf00, ctx00 = load_model_xform(cellid=site, modelname=model00, batch=331)


    # first order prediction -- what's the dimensionality of the residual?
    r0 = ctx['val']
    #r0 = r0.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)

    res0 = r0.apply_mask()['pred0']._data - r0.apply_mask()['psth_sp']._data 

    # do PCA and plot scree plot to check dimensionality
    pca = PCA()
    pca.fit(res0.T)
    f, ax = plt.subplots(1, 3, figsize=(9, 3))

    ax[0].plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
    ax[0].set_xlabel('PCs (first-order model residual)')
    ax[0].set_ylabel('Cum. fract. variance explained')
    ax[0].set_ylim((0, 1.1))

    # get LV axis
    lvax = ctx['modelspec'][0]['phi']['g'][:, 1]
    lvax /= np.linalg.norm(lvax)  # get unit vector for comparison with pcs

    # plot similarity with each first order pupil axis
    ax[0].plot(np.abs(lvax.dot(pca.components_.T)), '.-')

    # Look at shuffled pup dimension as controal
    lv0ax = ctx00['modelspec'][0]['phi']['g'][:, 1]
    lv0ax /= np.linalg.norm(lv0ax)
    ax[0].plot(np.abs(lv0ax.dot(pca.components_.T)), '.-')

    ax[0].set_title(site+f" num. neurons={r0['resp'].shape[0]}")

    # compare LV weights to first order weights
    ax[1].scatter(pca.components_[0], lvax, s=15)
    ax[1].axis('equal')
    ax[1].set_title(f"d={ctx['modelspec'][0]['phi']['d'][:, 1]}")

    # same for shuffled
    ax[2].scatter(pca.components_[0], lv0ax, s=15)
    ax[2].axis('equal')

    f.tight_layout()

plt.show()