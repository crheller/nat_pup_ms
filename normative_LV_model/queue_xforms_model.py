"""
Script for queuing xforms latent variable models
"""
import nems.db as nd

from global_settings import HIGHR_SITES, CPN_SITES

force_rerun = True
exacloud = True
stategain = True
gain = True
epochs = 'er5'

# all four models, ss1, 1sec mvm mask
# 30.09.2021 -- Noticed a bug in the epoch selection due to `.sp` option. Remove this going forward.
#modellist = [
#    "psth.fs4.pup-ld-st.pup0.pvp-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10.sp-aev_sdexp2.SxR-lvnorm.2xR.d.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
#    "psth.fs4.pup-ld-st.pup0.pvp0-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10.sp-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
#    "psth.fs4.pup-ld-st.pup0.pvp-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10.sp-aev_sdexp2.SxR-lvnorm.2xR.d.so-inoise.SxR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
#    "psth.fs4.pup-ld-st.pup.pvp0-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10.sp-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
#    "psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10.sp-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1"
#]

# 'new' models as of 30.09.2021 -- 
# use tolerance 1e-5, just for consistency with old models (loadpred days)
# don't use `.sp` (it doesn't do anything anymore, but just to differentiate new from old fits)
# try fitting models without the "extra" latent variables. e.g. indep becomes lvnorm.1xR instead of 2xR
# order of models: first order only, indep pup noise only, single pup lv, two pup lvs
modellist = [
    "psth.fs4.pup-ld-st.pup0.pvp-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10-aev_sdexp2.SxR-lvnorm.2xR.d.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
    "psth.fs4.pup-ld-st.pup0.pvp-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10-aev_sdexp2.SxR-lvnorm.2xR.d.so-inoise.SxR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
    "psth.fs4.pup-ld-st.pup.pvp0-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
    "psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1"
]
# add the pared down models (without extra lv signals, e.g. pvp0)
modellist += [
    "psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10-aev_sdexp2.SxR-lvnorm.1xR.d.so-inoise.1xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
    "psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10-aev_sdexp2.SxR-lvnorm.1xR.d.so-inoise.SxR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
    "psth.fs4.pup-ld-st.pup-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10-aev_sdexp2.SxR-lvnorm.SxR.d.so-inoise.SxR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1",
    "psth.fs4.pup-ld-st.pup.pvp-epcpn-mvm.t25.w1-hrc-psthfr.z-plgsm.e10-aev_sdexp2.2xR-lvnorm.SxR.d.so-inoise.2xR_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss1"
]

# add ss2 / ss3 / full rank
modellist += [m.replace('ss1', 'ss2') for m in modellist]
modellist += [m.replace('ss1', 'ss3') for m in modellist if 'ss1' in m]
modellist += [m.replace('.ss1', '') for m in modellist if 'ss1' in m]
# 2 sec movement mask
modellist += [m.replace('-mvm.t25.w1', '-mvm.t25.w2') for m in modellist]
# no movement mask
modellist += [m.replace('-mvm.t25.w1', '') for m in modellist if 'w1' in m]
# specify which epochs to fit using epochs variable
modellist = [m.replace('e10', epochs) for  m in modellist]

if stategain:
    modellist = [m.replace('sdexp2', 'stategain') for m in modellist]
    modellist = [m.replace('-lvnorm', '-spred-lvnorm') for m in modellist]

if gain:
    modellist = [m.replace('.d.so', '.so') for m in modellist]

# slow drift control models
# modellist = [m.replace('.pvp0', '').replace('.pvp', '').replace('st.pup0-', 'st.drf.pup0-').replace('st.pup-', 'st.drf.pup-') for m in modellist]

sites = CPN_SITES + HIGHR_SITES
batches = [331] * len(CPN_SITES) + [322]*len(HIGHR_SITES)

# manual code to pare down models that we fit for testing. This changes all the time
modellist = [m for m in modellist if ('mvm' not in m)]

if exacloud:
    from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models
    #force_rerun=True
    lbhb_user="hellerc"
    # exacloud settings:
    executable_path = '/home/users/davids/anaconda3/envs/nems/bin/python'  # works for GPU / non GPU
    script_path = '/home/users/davids/nems/scripts/fit_single.py'
    ssh_key = '/home/svd/.ssh/id_rsa'
    user = "davids"
    for s, b in zip(sites, batches):
        if s in ['BOL005c', 'BOL006b']:
            b = 294
        #if b == 331:
        #    if movement_mask:
        #        _modellist = [m.replace('loadpred', f'loadpred.{mask_key}') for m in modellist]
        #    else:
        #        _modellist = [m.replace('loadpred', 'loadpred.cpn') for m in modellist]
        #else:
        #    _modellist = modellist
        if b in [322, 294]:
            _modellist = [m.replace('epcpn-', '') for m in modellist]
        else:
            _modellist = modellist
        enqueue_exacloud_models(
            cellist=[s], batch=b, modellist=_modellist,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path, script_path=script_path, useGPU=False,
            time_limit=20)

else:
    # queue up a batch of jobs
    python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'
    python_path = '/auto/users/hellerc/anaconda3/envs/tf/bin/python'
    script = '/auto/users/hellerc/code/NEMS/scripts/fit_single.py'

    for s, b in zip(sites, batches):
        if s in ['BOL005c', 'BOL006b']:
            b = 294
        #if b == 331:
        #    if movement_mask:
        #        _modellist = [m.replace('loadpred', f'loadpred.{mask_key}') for m in modellist]
        #    else:
        #        _modellist = [m.replace('loadpred', 'loadpred.cpn') for m in modellist]
        #else:
        #    _modellist = modellist
        _modellist = modellist
        nd.enqueue_models(celllist=[s],
                        batch=b,
                        modellist=_modellist,
                        executable_path=python_path,
                        script_path=script,
                        user='hellerc',
                        force_rerun=force_rerun,
                        reserve_gb=4)



# old params for the loadpred models, where first order / second order were fit separately
'''
movement_mask = False # False
mask_key = 'cpnmvm,t25,w1' #'cpnmvm,t25,w1' # 'cpnOldmvm,t25,w1'
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
'''