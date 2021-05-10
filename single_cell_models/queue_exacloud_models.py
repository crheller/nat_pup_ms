"""
First order models for all cells simultaneously. Used to feed into LV models
"""
from global_settings import CPN_SITES

modelnames = [
    'psth.fs4.pup-ld-st.pup-epcpn-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr3e4.cont.et5.i50000',
    'psth.fs4.pup-ld-st.pup0-epcpn-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr3e4.cont.et5.i50000'
]

from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models
force_rerun = False
#force_rerun=True
lbhb_user="svd"
# exacloud settings:
executable_path = '/home/users/davids/anaconda3/envs/nems/bin/python'
script_path = '/home/users/davids/nems/scripts/fit_single.py'
ssh_key = '/home/svd/.ssh/id_rsa'
user = "davids"

batch = 331
sites = CPN_SITES
enqueue_exacloud_models(
    cellist=sites, batch=batch, modellist=modelnames,
    user=lbhb_user, linux_user=user, force_rerun=force_rerun,
    executable_path=executable_path, script_path=script_path, useGPU=True)