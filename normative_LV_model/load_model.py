"""
Load single model, extract weights / LVs etc.
Compare with true rec
Compare decoding
"""
from nems.xform_helper import load_model_xform

import matplotlib.pyplot as plt
import numpy as np

site = 'AMT020a'
batch = 331

modelname = "psth.fs4.pup-loadpred.cpn-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2"

xf, ctx = load_model_xform(site, batch, modelname=modelname)

# to run locally:
# xf1, ctx1 = fit_model_xform(cellid, batch, modelname, returnModel=True)
#import nems.plots.state as sp
#import numpy as np
#extra_epochs = np.unique([k.strip('mask_').strip('_sm').strip('_lg') for k in ctx1['val'].signals.keys() if ':' in k]).tolist()
#d=sp.cc_comp(ctx1['val'],ctx1['modelspec'], extra_epoch=extra_epochs)


# raw data look at example pairs
from global_settings import CPN_SITES
import charlieTools.nat_sounds_ms.decoding as decoding

# bad sites: AMT021b (no pupil variability), AMT005e (no pupil variability), CRD005b (no pupil variability), CRD019b (no pupil variability), 

site = 'ARM031a'
batch = 331

res = decoding.DecodingResults()
data = res.load_results(f"/auto/users/hellerc/results/nat_pupil_ms/dprime_final/{batch}/{site}/dprime_jk10_zscore_nclvz_fixtdr2_TDR.pickle")

df = data.numeric_results
df[['bp_dp', 'sp_dp', 'bp_dU_mag', 'sp_dU_mag']].head()
combo = (0, 11)
decoding.plot_stimulus_pair(site, batch, combo, pup_split=True, ellipse=True) #, ylim=(-5, 5), xlim=(-5, 5))
plt.axis('equal')
plt.show()

for site in CPN_SITES:

    res = decoding.DecodingResults()
    data = res.load_results(f"/auto/users/hellerc/results/nat_pupil_ms/dprime_final/{batch}/{site}/dprime_jk10_zscore_nclvz_fixtdr2_TDR.pickle")

    df = data.numeric_results

    f, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].scatter(df['bp_dU_mag'], df['sp_dU_mag'], s=10)
    ax[0].plot([0, 10], [0, 10], 'k--') 
    ax[0].set_xlabel("Large")
    ax[0].set_ylabel("Small")
    ax[0].set_title(r"$\Delta \mu$ ($\mu_a - \mu_b$)")

    ax[1].scatter(df['bp_dp'], df['sp_dp'], s=10)
    ax[1].plot([0, 100], [0, 100], 'k--') 
    ax[1].set_xlabel("Large")
    ax[1].set_ylabel("Small")
    ax[1].set_title(r"$d'^2$")

    f.canvas.set_window_title(site)

plt.show()
