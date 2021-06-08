"""
Compare delta dprime between models
"""

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

recache = False
sites = CPN_SITES
batches = [331]*len(CPN_SITES)

decoder = 'dprime_jk10_zscore_nclvz_fixtdr2-fa'
indep = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss3"
rlv = "psth.fs4.pup-loadpred.cpn-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.so-inoise.2xR_ccnorm.t6.ss3"
plv = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.so-inoise.2xR_ccnorm.t6.ss3"

for batch, site in zip(batches, sites): #[s for s in HIGHR_SITES if s not in ['CRD017c', 'CRD016d']]:
    if site in ['BOL006b', 'BOL005c']:
        batch = batch2 = 294

    loader = decoding.DecodingResults()
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+'_TDR.pickle')
    raw = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{rlv}_TDR.pickle')
    lv0 = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{indep}_TDR.pickle')
    indep = loader.load_results(fn, cache_path=None, recache=recache)
    fn = os.path.join(DPRIME_DIR, str(batch), site, decoder+f'_model-LV-{plv}_TDR.pickle')
    lv = loader.load_results(fn, cache_path=None, recache=recache)

    # save results for each model and for fit / not fit stimuli