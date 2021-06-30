"""
Script for queuing xforms latent variable models
"""
from nems.xform_helper import fit_model_xform
import nems.db as nd

from global_settings import HIGHR_SITES, CPN_SITES

movement_mask = True # False
epochs = 'e12'

# try to build a model string that fits first order state model + LV model in the same job
'psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w2-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000'
'psth.fs4.pup-ld-st.pup0.pvp0-epcpn-mvm.25.2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_init.ff1-ccnorm.r.t5.ss1'
'psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_init.ff1-ccnorm.r.t5.ss1'

# LV models
modellist = [
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss2",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss3",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss3",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d.so-inoise.2xR_ccnorm.t5.ss3"
]

# independent noise only model
modellist += [
    "psth.fs4.pup-loadpred-st.pup0.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss1",
    "psth.fs4.pup-loadpred-st.pup0.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss2",
    "psth.fs4.pup-loadpred-st.pup0.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.2xR.d.so-inoise.3xR_ccnorm.t5.ss3"
]

# testing new...
modellist = [
    "psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_init.xx1.it5000-ccnorm.f0.ss1",
    "psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_init.xx1.it5000",
    "psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w2-hrc-psthfr-plgsm.e10.sp-lvnoise.r8-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_init.xx1.it5000-ccnorm.f0.ss1-ccnorm.r.ss1"
]
##########################3

# replace 'good' epoch fits with random subset for more fair validation metric
modellist = [m.replace('eg10', epochs) for m in modellist]
#modellist = [m.replace('e10', 'e5') for m in modellist]

# slow drift control models
# modellist = [m.replace('.pvp0', '').replace('.pvp', '').replace('st.pup0-', 'st.drf.pup0-').replace('st.pup-', 'st.drf.pup-') for m in modellist]

# queue up a batch of jobs
force_rerun = True
python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'
script = '/auto/users/hellerc/code/NEMS/scripts/fit_single.py'

sites = CPN_SITES #+ HIGHR_SITES
batches = [331] * len(CPN_SITES) #+ [322]*len(HIGHR_SITES)
#sites = HIGHR_SITES
#batches = [322] * len(HIGHR_SITES)

for s, b in zip(sites, batches):
    if s in ['BOL005c', 'BOL006b']:
        b = 294
    if b == 331:
        if movement_mask:
            _modellist = [m.replace('loadpred', 'loadpred.cpnmvm') for m in modellist]
        else:
            _modellist = [m.replace('loadpred', 'loadpred.cpn') for m in modellist]
    else:
        _modellist = modellist
    nd.enqueue_models(celllist=[s],
                      batch=b,
                      modellist=_modellist,
                      executable_path=python_path,
                      script_path=script,
                      user='hellerc',
                      force_rerun=force_rerun,
                      reserve_gb=4)