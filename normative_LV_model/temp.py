"""
Load / analyse single LV model fit.
Look at "good" stimulus pairs and "bad" stimulus pairs.
"""
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, PY_FIGURES_DIR2, CACHE_PATH, REGRESSION
import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import numpy as np

from nems.xform_helper import load_model_xform

load = True
sites = ['DRX008b.e65:128'] #HIGHR_SITES #'TAR010c'
for site in sites: #[s for s in HIGHR_SITES if s not in ['CRD017c', 'CRD016d']]:
    if site in ['BOL006b', 'BOL005c']:
        batch = batch2 = 294
    else:
        batch = 322; batch2 = 289
    modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5.ss3"

    if load:
        xf, ctx = load_model_xform(modelname=modelname, batch=batch, cellid=site)
        rec = ctx['val']

    decoder = 'dprime_jk10_zscore_nclvz_fixtdr2'
    recache = True
    loader = decoding.DecodingResults()
    fn = os.path.join(DPRIME_DIR, site, decoder+'_TDR.pickle')
    dpres = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    fn = os.path.join(DPRIME_DIR, site, decoder+f'_model-LV-{modelname}_TDR.pickle')
    lvres = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    #fn = os.path.join(DPRIME_DIR, site, 'dprime_simInTDR_sim12_jk10_zscore_nclvz_fixtdr2'+'_TDR.pickle')
    fn = fn.replace('st.pup.pvp', 'st.pup0.pvp0')
    lvres0 = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)

    # get the epochs of interest (fit epochs)
    mask_bins = lvres.meta['mask_bins']
    # map these to the dataframe combos
    fb_combos = [k for k, v in lvres.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                    (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]

    # get the true / predicted deltas
    rawdf = dpres.numeric_results.loc[fb_combos]
    lvdf = lvres.numeric_results.loc[fb_combos]
    lv0df = lvres0.numeric_results.loc[fb_combos]

    raw_delta = (rawdf['bp_dp'] - rawdf['sp_dp']) / (rawdf['bp_dp'] + rawdf['sp_dp'])
    lv_delta = (lvdf['bp_dp'] - lvdf['sp_dp']) / (lvdf['bp_dp'] + lvdf['sp_dp'])
    lv0_delta = (lv0df['bp_dp'] - lv0df['sp_dp']) / (lv0df['bp_dp'] + lv0df['sp_dp'])

    f, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

    ax[0].scatter(raw_delta, lv_delta)
    ax[0].set_xlabel('Raw delta')
    ax[0].set_ylabel('Pred. delta')
    ax[0].set_title('r: {0}'.format(np.round(np.corrcoef(raw_delta, lv_delta)[0, 1], 3)))
    ax[0].axhline(0, linestyle='--', color='k')
    ax[0].axvline(0, linestyle='--', color='k')


    ax[1].scatter(raw_delta, lv0_delta)
    ax[1].set_xlabel('Raw delta')
    ax[1].set_ylabel('Pred. delta (shuffle LV)')
    ax[1].set_title('r: {0}'.format(np.round(np.corrcoef(raw_delta, lv0_delta)[0, 1], 3)))
    ax[1].axhline(0, linestyle='--', color='k')
    ax[1].axvline(0, linestyle='--', color='k')

    f.tight_layout()

    f.canvas.set_window_title(site)

    # now, find a couple cases where the diff in the delta is big between predicted and raw results
    idx = np.argsort(abs(raw_delta - lv_delta))[::-1]
    big_diff = idx[:3].index.get_level_values(0)

    e1, e2 = lvres.mapping[big_diff[0]]

    

plt.show()
