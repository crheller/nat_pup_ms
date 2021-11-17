"""
Pop model for single cell
"""
import nems.db as nd
from global_settings import CPN_SITES, HIGHR_SITES

batch = 294 # 294, 289 || 323 (PEG) | 331 (CPN)
batch = 331
force_rerun = True
exacloud = True
sites = CPN_SITES + HIGHR_SITES
batches = [331]*len(CPN_SITES) + [322]*len(HIGHR_SITES)


# 04/26/2020 - Queue sdexp models again. New sdexp architecture allows easy extraction 
# gain params. Should be easy to invert these models to get rid of first order pupil
modelnames = ['psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr-aev_stategain.SxR_tfinit.n.lr1e4.cont.et5.i50000',
              'psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr-aev_sdexp2.SxR_tfinit.n.lr1e4.cont.et5.i50000',
              'psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr_stategain.SxR_jk.nf10-tfinit.n.lr1e4.cont.et5.i50000',
              'psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr_sdexp2.SxR_jk.nf10-tfinit.n.lr1e4.cont.et5.i50000'
                ]


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