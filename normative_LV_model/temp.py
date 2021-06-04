"""
Load / analyse single LV model fit.
Look at "good" stimulus pairs and "bad" stimulus pairs.
"""
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT, CPN_SITES
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, PY_FIGURES_DIR2, CACHE_PATH, REGRESSION
import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import numpy as np
import os

from dDR.dDR import dDR

from nems.xform_helper import load_model_xform

load = False
sites = ['DRX008b.e65:128'] #HIGHR_SITES #'TAR010c'
sites = CPN_SITES
#sites = HIGHR_SITES
batches = [331]*len(CPN_SITES)
#batches = [289]*len(HIGHR_SITES)

decoder = 'dprime_jk10_zscore_nclvz_fixtdr2'
#decoder = 'dprime_pca-4-psth-whiten_jk10_nclvz_fixtdr2'
#decoder = 'dprime_pca-4-psth-whiten_jk10_nclvz_fixtdr2'
norm_diff = True
plot_individual = True

modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5.ss2"
modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2"
modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3"
modelname = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2"
#modelname = "psth.fs4.pup-loadpred.pc4-st.pup.pvp-plgsm.e-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_pcnorm.t6"
#modelname = "psth.fs4.pup-loadpred.pc4-st.pup.pvp-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_pcnorm.t6"
#modelname = "psth.fs4.pup-loadpred.pc4-st.pup.pvp-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_pcnorm.t6"
#modelname = "psth.fs4.pup-loadpred.pc4-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6"

all_raw = []
all_lv = []
all_lv0 = []
bad_flags = []
for batch, site in zip(batches, sites): #[s for s in HIGHR_SITES if s not in ['CRD017c', 'CRD016d']]:
    if site in ['BOL006b', 'BOL005c']:
        batch = batch2 = 294
    if batch == 289:
        batch2 = 289
        batch = 289
    elif batch == 331:
        batch = batch2 = 331

    if load:
        xf, ctx = load_model_xform(modelname=modelname, batch=batch, cellid=site)
        rec = ctx['val']

    recache = True
    loader = decoding.DecodingResults()
    fn = os.path.join(DPRIME_DIR, str(batch2), site, decoder+'_TDR.pickle')
    dpres = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{modelname}_TDR.pickle')
    lvres = loader.load_results(fn, cache_path=None, recache=recache)
    #fn = os.path.join(DPRIME_DIR, str(batch), site, 'dprime_simInTDR_sim12_jk10_zscore_nclvz_fixtdr2'+'_TDR.pickle')
    fn = fn.replace('st.pup.pvp', 'st.pup.pvp0')
    lvres0 = loader.load_results(fn, cache_path=None, recache=recache)

    if 'jk10' in decoder:
        cvres = dpres
    else:
        fn = os.path.join(DPRIME_DIR, str(batch), site, decoder.replace('jk1_eev', 'jk10')+'_TDR.pickle')
        cvres = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)

    # get the epochs of interest (fit epochs)
    try:
        mask_bins = lvres.meta['mask_bins']
        # map these to the dataframe combos
        fb_combos = [k for k, v in lvres.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                        (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]
        s = 25
    except:
        fb_combos = lvres.numeric_results.index.get_level_values(0)
        s = 5

    # get the true / predicted deltas
    rawdf = dpres.numeric_results.loc[fb_combos]
    lvdf = lvres.numeric_results.loc[fb_combos]
    lv0df = lvres0.numeric_results.loc[fb_combos]

    if norm_diff:
        raw_delta = (rawdf['bp_dp'] - rawdf['sp_dp']) / (rawdf['bp_dp'] + rawdf['sp_dp'])
        lv_delta = (lvdf['bp_dp'] - lvdf['sp_dp']) / (lvdf['bp_dp'] + lvdf['sp_dp'])
        lv0_delta = (lv0df['bp_dp'] - lv0df['sp_dp']) / (lv0df['bp_dp'] + lv0df['sp_dp'])
    else:
        raw_delta = (rawdf['bp_dp'] - rawdf['sp_dp'])
        lv_delta = (lvdf['bp_dp'] - lvdf['sp_dp']) 
        lv0_delta = (lv0df['bp_dp'] - lv0df['sp_dp'])

    if plot_individual:
        f, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

        ax[0].scatter(raw_delta, lv_delta, s=s)
        ax[0].set_xlabel('Raw delta')
        ax[0].set_ylabel('Pred. delta')
        ax[0].set_title('r: {0}'.format(np.round(np.corrcoef(raw_delta, lv_delta)[0, 1], 3)))
        ax[0].axhline(0, linestyle='--', color='k')
        ax[0].axvline(0, linestyle='--', color='k')

        ax[1].scatter(raw_delta, lv0_delta, s=s)
        ax[1].set_xlabel('Raw delta')
        ax[1].set_ylabel('Pred. delta (shuffle LV)')
        ax[1].set_title('r: {0}'.format(np.round(np.corrcoef(raw_delta, lv0_delta)[0, 1], 3)))
        ax[1].axhline(0, linestyle='--', color='k')
        ax[1].axvline(0, linestyle='--', color='k')

        f.tight_layout()

        f.canvas.set_window_title(site)

    # now, find a couple cases where the diff in the delta is big between predicted and raw results
    idx = np.argsort(abs(raw_delta - lv_delta))[::-1]
    big_diff = [raw_delta.index.get_level_values(0)[idx[2]]]

    e1, e2 = lvres.mapping[big_diff[0]]
    e1, b1 = ('_'.join(e1.split('_')[:-1]), int(e1.split('_')[-1]))
    e2, b2 = ('_'.join(e2.split('_')[:-1]), int(e2.split('_')[-1]))

    all_raw.extend(list(raw_delta.values))
    all_lv.extend(list(lv_delta.values))
    all_lv0.extend(list(lv0_delta.values))

    # bad flags for cases with unreliable noise estimates
    bfs = [True] * len(raw_delta)
    varall = cvres.array_results['evecs_test']['sem'].loc[:,2].apply(lambda x: np.sqrt(sum(x[:,0]**2)))
    var = cvres.array_results['evecs_test']['sem'].loc[fb_combos,2].apply(lambda x: np.sqrt(sum(x[:,0]**2)))
    #bad_flags.extend(list((var>=(2*np.std(varall) + np.median(varall))).values))
    bad_flags.extend(list((var>=0.05).values))

    if load:
        # plot pred responses in dDR space (use first PC of real data as "noise" axis to approximate true ddr space)
        noise_axis = dpres.array_results['evecs_all'].loc[big_diff[0]]['mean'].values[0][:, [0]].T
        noise_axis = noise_axis / np.linalg.norm(noise_axis)
        r1 = rec['resp'].extract_epoch(e1)[:, :, b1]
        r2 = rec['resp'].extract_epoch(e2)[:, :, b2]
        p1 = rec['pred'].extract_epoch(e1)[:, :, b1]
        p2 = rec['pred'].extract_epoch(e2)[:, :, b2]
        pup1m = rec['pupil'].extract_epoch(e1)[:, :, b1]
        pup1m = pup1m.squeeze() >= np.median(pup1m)
        pup2m = rec['pupil'].extract_epoch(e2)[:, :, b2]
        pup2m = pup2m.squeeze() >= np.median(pup2m)    
        ddr = dDR(ddr2_init=noise_axis)
        dr1, dr2 = ddr.fit_transform(r1, r2)
        dp1 = ddr.transform(p1)
        dp2 = ddr.transform(p2)

        f, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

        ax[0].scatter(dr1[~pup1m, 0], dr1[~pup1m, 1], color='tab:blue')
        ax[0].scatter(dr1[pup1m, 0], dr1[pup1m, 1], facecolor='white', color='tab:blue')
        ax[0].scatter(dr2[~pup2m, 0], dr2[~pup2m, 1], color='tab:orange')
        ax[0].scatter(dr2[pup2m, 0], dr2[pup2m, 1], facecolor='white', color='tab:orange')
        ax[0].set_title(r"Real data, $\Delta d'^2=%s$" % round(raw_delta.loc[big_diff[0]].values[0], 3))

        ax[1].scatter(dp1[~pup1m, 0], dp1[~pup1m, 1], color='tab:blue')
        ax[1].scatter(dp1[pup1m, 0], dp1[pup1m, 1], facecolor='white', color='tab:blue')
        ax[1].scatter(dp2[~pup2m, 0], dp2[~pup2m, 1], color='tab:orange')
        ax[1].scatter(dp2[pup2m, 0], dp2[pup2m, 1], facecolor='white', color='tab:orange')
        ax[1].set_title(r"Pred data, $\Delta d'^2=%s$" % round(lv_delta.loc[big_diff[0]].values[0], 3))

        f.tight_layout()
bad_flags = np.array(bad_flags)
f, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)

s = 5
# bad pairs
ax[0].scatter(np.array(all_raw)[bad_flags], np.array(all_lv)[bad_flags], s=s, label="'bad' pairs", color='tab:blue', alpha=0.3)
m, b = np.polyfit(np.array(all_raw)[bad_flags], np.array(all_lv)[bad_flags], 1)
ax[0].plot(np.array(all_raw)[bad_flags], m*np.array(all_raw)[bad_flags] + b, lw=2, color='tab:blue')
# good pairs
ax[0].scatter(np.array(all_raw)[~bad_flags], np.array(all_lv)[~bad_flags], label="'good' pairs", s=s+5, color='tab:orange', alpha=0.8)
m, b = np.polyfit(np.array(all_raw)[~bad_flags], np.array(all_lv)[~bad_flags], 1)
ax[0].plot(np.array(all_raw)[~bad_flags], m*np.array(all_raw)[~bad_flags] + b, lw=2, color='tab:orange')

ax[0].set_xlabel('Raw delta')
ax[0].set_ylabel('Pred. delta')
ax[0].set_title('r good: {0}, r bad: {1}'.format(np.round(np.corrcoef(np.array(all_raw)[~bad_flags], np.array(all_lv)[~bad_flags])[0, 1], 3),
                                                np.round(np.corrcoef(np.array(all_raw)[bad_flags], np.array(all_lv)[bad_flags])[0, 1], 3)))
ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0, linestyle='--', color='k')
ax[0].legend()

# bad pairs
ax[1].scatter(np.array(all_raw)[bad_flags], np.array(all_lv0)[bad_flags], s=s, label="'bad' pairs", color='tab:blue', alpha=0.3)
m, b = np.polyfit(np.array(all_raw)[bad_flags], np.array(all_lv0)[bad_flags], 1)
ax[1].plot(np.array(all_raw)[bad_flags], m*np.array(all_raw)[bad_flags] + b, lw=2, color='tab:blue')
# good pairs
ax[1].scatter(np.array(all_raw)[~bad_flags], np.array(all_lv0)[~bad_flags], label="'good' pairs", s=s+5, color='tab:orange', alpha=0.8)
m, b = np.polyfit(np.array(all_raw)[~bad_flags], np.array(all_lv0)[~bad_flags], 1)
ax[1].plot(np.array(all_raw)[~bad_flags], m*np.array(all_raw)[~bad_flags] + b, lw=2, color='tab:orange')

ax[1].set_xlabel('Raw delta')
ax[1].set_ylabel('Pred. delta (shuffle LV)')
ax[1].set_title('r good: {0}, r bad: {1}'.format(np.round(np.corrcoef(np.array(all_raw)[~bad_flags], np.array(all_lv0)[~bad_flags])[0, 1], 3),
                                                np.round(np.corrcoef(np.array(all_raw)[bad_flags], np.array(all_lv0)[bad_flags])[0, 1], 3)))
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0, linestyle='--', color='k')

f.suptitle(modelname, fontsize=6)

f.tight_layout()

f.canvas.set_window_title('All data')

plt.show()
