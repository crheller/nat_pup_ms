'''
Latent variable model results

Goal -- show that low-dimensional latent variable model captures the variety of decoding effects

Ideas (excluding the schematic - to be made in inkscape):
    1) Example site
        - big/small corr matrices for indep. noise model, random LV model, pupil-dependent LV, and true data
        - Example ellipse plots for each of these models for a single stimulus pair
    2) Summary across sites
        - Delta dprime scatter plots for each model (highlight fit / not fit stims)
    3) Example pairs
        - Highlight two example pairs on the scatter plot (upper right / lower left) and 
            show their ellipse plots. Pick a site where noise corr. definitely decreases
            and have it correspond with the example site where we're showing corr. matrices
'''
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT, CPN_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, PY_FIGURES_DIR2, CACHE_PATH, REGRESSION
import charlieTools.nat_sounds_ms.decoding as decoding
import figures_final.helpers as fhelp

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 6

from nems.xform_helper import load_model_xform

site = 'AMT020a'
batch = 331

# LOAD RAW DATA / MODEL PREDICTIONS
indep = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss3"
rlv = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss3"
plv = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss3"

xf_indep, ctx_indep = load_model_xform(modelname=indep, batch=batch, cellid=site)
xf_rlv, ctx_rlv = load_model_xform(modelname=rlv, batch=batch, cellid=site)
xf_plv, ctx_plv = load_model_xform(modelname=plv, batch=batch, cellid=site)

# GET COV MATRICES
ibg, ism = fhelp.get_cov_matrices(ctx_indep['val'].copy(), sig='pred')
rbg, rsm = fhelp.get_cov_matrices(ctx_rlv['val'].copy(), sig='pred')
pbg, psm = fhelp.get_cov_matrices(ctx_plv['val'].copy(), sig='pred')
bg, sm = fhelp.get_cov_matrices(ctx_plv['val'].copy(), sig='resp')
mm = np.abs(np.max(np.concatenate((ibg, ism, rbg, rsm, pbg, psm, bg, sm))))
mm=10
dmm=5
f, ax = plt.subplots(3, 4, figsize=(8, 6))

ax[0, 0].set_title('Pupil-independent (st.pup0.pvp0)')
ax[0, 0].set_ylabel('BIG PUPIL')
ax[0, 0].imshow(rbg, aspect='auto', vmin=-mm, vmax=mm, cmap='bwr')
ax[1, 0].imshow(rsm, aspect='auto', vmin=-mm, vmax=mm, cmap='bwr')
ax[1, 0].set_ylabel('SMALL PUPIL')
ax[2, 0].imshow(rsm-rbg, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')
ax[2, 0].set_ylabel('DIFF')

ax[0, 1].set_title('Pupil-dependent indep. noise')
ax[0, 1].imshow(ibg, aspect='auto', vmin=-mm, vmax=mm, cmap='bwr')
ax[1, 1].imshow(ism, aspect='auto', vmin=-mm, vmax=mm, cmap='bwr')
ax[2, 1].imshow(ism-ibg, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')

ax[0, 2].set_title('Pupil-dependent LV (st.pup.pvp)')
ax[0, 2].imshow(pbg, aspect='auto', vmin=-mm, vmax=mm, cmap='bwr')
ax[1, 2].imshow(psm, aspect='auto', vmin=-mm, vmax=mm, cmap='bwr')
ax[2, 2].imshow(psm-pbg, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')

ax[0, 3].set_title('Raw data')
ax[0, 3].imshow(bg, aspect='auto', vmin=-mm, vmax=mm, cmap='bwr')
ax[1, 3].imshow(sm, aspect='auto', vmin=-mm, vmax=mm, cmap='bwr')
ax[2, 3].imshow(sm-bg, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')

f.tight_layout()

plt.show()
