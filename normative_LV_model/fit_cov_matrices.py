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

site = 'AMT020a'
batch = 331

# LOAD RAW DATA / MODEL PREDICTIONS
indep = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss1"
rlv = "psth.fs4.pup-loadpred.cpnmvm-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"
plv = "psth.fs4.pup-loadpred.cpnmvm-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1"

reverything = 'psth.fs4.pup-ld-st.pup0.pvp0-epcpn-mvm.25.2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_ccnorm.r.t5.ss1'
indep = 'psth.fs4.pup-ld-st.pup0.pvp-epcpn-mvm.25.2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.2xR.d.so-inoise.3xR_ccnorm.r.t5.ss1'
rlv = 'psth.fs4.pup-ld-st.pup0.pvp-epcpn-mvm.25.2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.2xR.d.so-inoise.2xR_ccnorm.r.t5.ss1'
plv = 'psth.fs4.pup-ld-st.pup.pvp0-epcpn-mvm.25.2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_ccnorm.r.t5.ss1'

try:
    cellid = site
    xf_indep, ctx_indep = load_model_xform(modelname=indep, batch=batch, cellid=cellid)
    xf_rlv, ctx_rlv = load_model_xform(modelname=rlv, batch=batch, cellid=cellid)
    xf_plv, ctx_plv = load_model_xform(modelname=plv, batch=batch, cellid=cellid)
except:
    cellid = [c for c in nd.get_batch_cells(batch).cellid if site in c][0]
    xf_indep, ctx_indep = load_model_xform(modelname=indep, batch=batch, cellid=cellid)
    xf_rlv, ctx_rlv = load_model_xform(modelname=rlv, batch=batch, cellid=cellid)
    xf_plv, ctx_plv = load_model_xform(modelname=plv, batch=batch, cellid=cellid)

# GET COV MATRICES
stim = np.arange(10)
ibg, ism = fhelp.get_cov_matrices(ctx_indep['val'].copy(), sig='pred', sub='psth_sp', stims=stim, ss=0)
rbg, rsm = fhelp.get_cov_matrices(ctx_rlv['val'].copy(), sig='pred', sub='psth_sp', stims=stim, ss=0)
pbg, psm = fhelp.get_cov_matrices(ctx_plv['val'].copy(), sig='pred', sub='psth_sp', stims=stim, ss=0)
bg, sm = fhelp.get_cov_matrices(ctx_plv['val'].copy(), sig='resp', sub='psth_sp', stims=stim, ss=0)
mm = np.abs(np.max(np.concatenate((ibg, ism, rbg, rsm, pbg, psm, bg, sm))))
mm=10
dmm=2

# LOAD DECODING RESULTS FOR MODEL / RAW DATA
recache = False
sites = CPN_SITES
batches = [331]*len(CPN_SITES)
decoder = 'dprime_mvm-25-2_jk10_zscore_nclvz_fixtdr2-fa_noiseDim-6'
results = {
    'fit': {
        'pup_indep': [],
        'indep_noise': [],
        'lv': [],
        'raw': []
    },
    'val': {
        'pup_indep': [],
        'indep_noise': [],
        'lv': [],
        'raw': []
    }
}
for batch, site in zip(batches, sites): #[s for s in HIGHR_SITES if s not in ['CRD017c', 'CRD016d']]:
    if site in ['BOL006b', 'BOL005c']:
        batch2 = 294
    else:
        batch2 = batch

    if batch in [289, 294]:
        _rlv = rlv.replace('.cpn', '')
        _ind = indep.replace('.cpn', '')
        _plv = plv.replace('.cpn', '')
    else:
        _rlv = rlv
        _ind = indep
        _plv = plv

    loader = decoding.DecodingResults()
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+'_TDR.pickle')
    raw = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+f'_model-LV-{_rlv}_TDR.pickle')
    lv0 = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+f'_model-LV-{_ind}_TDR.pickle')
    indep_noise = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+f'_model-LV-{_plv}_TDR.pickle')
    lv = loader.load_results(fn, cache_path=None, recache=recache)

    # get the epochs of interest (fit epochs)
    mask_bins = lv.meta['mask_bins']
    fit_combos = [k for k, v in lv.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                        (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]
    all_combos = lv.evoked_stimulus_pairs
    val_combos = [c for c in all_combos if c not in fit_combos]

    # save results for each model and for divide by fit / not fit stimuli
    
    # fit stims first
    for k, res in zip(['pup_indep', 'indep_noise', 'lv', 'raw'], [lv0, indep_noise, lv, raw]):
        df = res.numeric_results
        df['delta_dprime'] = (df['bp_dp'] - df['sp_dp']) #/ (df['bp_dp'] + df['sp_dp'])#(raw.numeric_results['bp_dp'] + raw.numeric_results['sp_dp']) #(df['bp_dp'] + df['sp_dp'])
        df['site'] = site
        results['fit'][k].append(df.loc[fit_combos])
        results['val'][k].append(df.loc[val_combos])

# concatenate data frames
for k in results['fit'].keys():
    results['fit'][k] = pd.concat(results['fit'][k])
    results['val'][k] = pd.concat(results['val'][k])



# MAKE FIGURE
dmmr = 2
dmm = 2

f, ax = plt.subplots(1, 4, figsize=(12, 3))

ax[0].set_title('Pupil-independent (st.pup0.pvp0)')
ax[0].set_ylabel('BIG PUPIL')
ax[0].set_ylabel('SMALL PUPIL')
ax[0].imshow(rsm-rbg, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')
ax[0].set_ylabel('DIFF')

ax[1].set_title('Independent noise')
ax[1].imshow(ism-ibg, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')

ax[2].set_title('Pupil-dependent LV (st.pup.pvp0)')
ax[2].imshow(psm-pbg, aspect='auto', vmin=-dmm, vmax=dmm, cmap='bwr')

ax[3].set_title('Raw data')
ax[3].imshow(sm-bg, aspect='auto', vmin=-dmmr, vmax=dmmr, cmap='bwr')

f.tight_layout()

plt.show()
