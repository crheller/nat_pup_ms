import nems.db as nd
from global_settings import HIGHR_SITES, CPN_SITES

batches = [294, 322, 331]
modelnames = ["factor_analysis"]
force_rerun = True

for batch in batches:

    if batch == 322:
        sites = [s for s in HIGHR_SITES if s not in ['BOL005c', 'BOL006b']]  
    if batch == 294:
        sites = ['BOL005c', 'BOL006b']
    if batch == 331:
        sites = CPN_SITES

    script = '/auto/users/hellerc/code/projects/nat_pupil_ms/FA/cache_population_stats.py'
    python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'
    nd.enqueue_models(celllist=sites,
                    batch=batch,
                    modellist=modelnames,
                    executable_path=python_path,
                    script_path=script,
                    user='hellerc',
                    force_rerun=force_rerun,
                    reserve_gb=2,
                    priority=2)