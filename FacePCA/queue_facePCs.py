import nems.db as nd

batches = [294, 322, 331]
modelnames = ["face_pca"]
force_rerun = True

for batch in batches:
    sites, _ = nd.get_batch_sites(batch)
    if batch==294:
        sites = [s for s in sites if s in ["BOL005c", "BOL006b"]]
    script = '/auto/users/hellerc/code/projects/nat_pupil_ms/FacePCA/cache_facePCs.py'
    python_path = '/auto/users/hellerc/miniconda3/envs/lbhb/bin/python'
            
    nd.enqueue_models(celllist=sites,
                    batch=batch,
                    modellist=modelnames,
                    executable_path=python_path,
                    script_path=script,
                    user='hellerc',
                    force_rerun=force_rerun,
                    reserve_gb=35)