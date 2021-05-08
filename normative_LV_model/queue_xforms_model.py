"""
Script for queuing xforms latent variable models
"""
from nems.xform_helper import fit_model_xform
import nems.db as nd

from global_settings import HIGHR_SITES

cellid='TAR010c'
batch=322
# 2 x LV
modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.e-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6"
# independent noise only
modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.e-lvnoise.r8-aev_inoise.2xR_ccnorm.t6"

modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg5.sp-lvnoise.r8-aev_inoise.2xR_ccnorm.t6"
modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6"
modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2"
modelname = "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2"

modellist = [
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3"
]

# to run locally:
# xf1, ctx1 = fit_model_xform(cellid, batch, modelname, returnModel=True)
import nems.plots.state as sp
import numpy as np
extra_epochs = np.unique([k.strip('mask_').strip('_sm').strip('_lg') for k in ctx1['val'].signals.keys() if ':' in k]).tolist()
d=sp.cc_comp(ctx1['val'],ctx1['modelspec'], extra_epoch=extra_epochs)

# queue up a batch of jobs
force_rerun = True
python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'
script = '/auto/users/svd/python/nems/scripts/fit_single.py'

sites = HIGHR_SITES
batches = [322 if s not in ['BOL006b', 'BOL005c'] else 294 for s in sites]
for s, b in zip(sites, batches):
    nd.enqueue_models(celllist=[s],
                      batch=b,
                      modellist=modellist,
                      executable_path=python_path,
                      script_path=script,
                      user='hellerc',
                      force_rerun=force_rerun,
                      reserve_gb=4)