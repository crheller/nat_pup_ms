"""
First order models for all cells simultaneously. Used to feed into LV models
"""
from global_settings import CPN_SITES

exacloud = False
batch = 331
sites = CPN_SITES
modelnames = [
    'psth.fs4.pup-ld-st.pup-epcpn-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000',
    'psth.fs4.pup-ld-st.pup0-epcpn-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000',
    'psth.fs4.pup-ld-st.pup-epcpn-mvm-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000',
    'psth.fs4.pup-ld-st.pup0-epcpn-mvm-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000'
]


if exacloud:
    from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models
    force_rerun = True
    #force_rerun=True
    lbhb_user="svd"
    # exacloud settings:
    executable_path = '/home/users/davids/anaconda3/envs/nems/bin/python'
    script_path = '/home/users/davids/nems/scripts/fit_single.py'
    ssh_key = '/home/svd/.ssh/id_rsa'
    user = "davids"

    enqueue_exacloud_models(
        cellist=sites, batch=batch, modellist=modelnames,
        user=lbhb_user, linux_user=user, force_rerun=force_rerun,
        executable_path=executable_path, script_path=script_path, useGPU=True)

else:
    script = '/auto/users/hellerc/code/NEMS/scripts/fit_single.py'
    python_path = '/auto/users/hellerc/anaconda3/envs/tf/bin/python'
    nd.enqueue_models(celllist=sites,
                  batch=batch,
                  modellist=modelnames,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  GPU_job=1)