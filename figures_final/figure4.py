'''
Latent variable model fits

Load latent variable model. Show predicted / actual covariance matrices 
(and rank one covariance matrix, since this is what was fit)

Stack in a vertical row to be show next to LV model schematic
'''
from path_settings import DPRIME_DIR, PY_FIGURES_DIR3, CACHE_PATH
import charlieTools.nat_sounds_ms.decoding as decoding
import figures_final.helpers as fhelp

import nems.db as nd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 6

from nems.xform_helper import load_model_xform

site = 'AMT020a'  # stims 4 / 8 are good examples
stim = 8  # which stim (or stims) to compute covariance matrix for, since model fit per stimulus
savefig = False
fig_fn = PY_FIGURES_DIR3 + 'fig4.svg'
plot_bg_sm = True

batch = 331
decoder = 'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-6'

rlv = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
indep = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss1"
plv = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"

try:
    cellid = site
    xf_plv, ctx_plv = load_model_xform(modelname=plv, batch=batch, cellid=cellid)
except:
    cellid = [c for c in nd.get_batch_cells(batch).cellid if site in c][0]
    xf_plv, ctx_plv = load_model_xform(modelname=plv, batch=batch, cellid=cellid)

# GET COV MATRICES
pbg, psm = fhelp.get_cov_matrices(ctx_plv['val'].copy(), sig='pred', sub='pred0', stims=stim, ss=0)
bg, sm = fhelp.get_cov_matrices(ctx_plv['val'].copy(), sig='resp', sub='pred0', stims=stim, ss=0)
bgr1, smr1 = fhelp.get_cov_matrices(ctx_plv['val'].copy(), sig='resp', sub='pred0', stims=stim, ss=1)

############################# MAKE FIGURE #############################
dmmr = np.max(np.concatenate((np.abs(sm-bg))))
dmm = dmmr

if not plot_bg_sm:
    f, ax = plt.subplots(3, 1, figsize=(2, 6))

    ax[0].set_title('Predicted')
    ax[0].imshow(psm-pbg, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')

    ax[1].set_title('Rank-1')
    ax[1].imshow(smr1-bgr1, aspect='auto', vmin=-dmmr, vmax=dmmr, cmap='bwr')

    ax[2].set_title('Full-rank')
    ax[2].imshow(sm-bg, aspect='auto', vmin=-dmmr, vmax=dmmr, cmap='bwr')

    for a in ax:
        a.set_yticks([])
        a.set_xticks([])

    f.tight_layout()

    if savefig:
        f.savefig(fig_fn)

else:
    dmm = 4
    f, ax = plt.subplots(3, 2, figsize=(4, 6))

    ax[0, 0].set_title('Predicted, Large')
    ax[0, 0].imshow(pbg, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')

    ax[0, 1].set_title('Predicted, Small')
    ax[0, 1].imshow(psm, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')

    ax[1, 0].set_title('Rank-1, Large')
    ax[1, 0].imshow(bgr1, aspect='auto', vmin=-dmmr, vmax=dmmr, cmap='bwr')

    ax[1, 1].set_title('Rank-1, Small')
    ax[1, 1].imshow(smr1, aspect='auto', vmin=-dmmr, vmax=dmmr, cmap='bwr')

    ax[2, 0].set_title('Full-rank, Large')
    ax[2, 0].imshow(bg, aspect='auto', vmin=-dmmr, vmax=dmmr, cmap='bwr')

    ax[2, 1].set_title('Full-rank, Small')
    ax[2, 1].imshow(sm, aspect='auto', vmin=-dmmr, vmax=dmmr, cmap='bwr')

    for a in ax.flatten():
        a.set_yticks([])
        a.set_xticks([])

    f.tight_layout()

    if savefig:
        f.savefig(fig_fn.replace('.svg', '_bgsm.svg'))


plt.show()