import nems.db as nd

batch = 289
force_rerun = True


modelnames = ['glm_ev_bal_nLV1_astep0.02_amax1',
              'glm_ev_nLV1_astep0.02_amax1',
              'glm_ev_bal_nLV2_astep0.02_amax1',
              'glm_ev_nLV2_astep0.02_amax1']
modelnames = ['glm_ev_bal_nLV3_astep0.02_amax1',
              'glm_ev_nLV3_astep0.02_amax1',
              'glm_ev_bal_nLV4_astep0.02_amax1',
              'glm_ev_nLV4_astep0.02_amax1',
              'glm_ev_bal_nLV5_astep0.02_amax1',
              'glm_ev_nLV5_astep0.02_amax1']
modelnames = ['glm_ev_bal_nLV0_astep0.02_amax1',
              'glm_ev_nLV0_astep0.02_amax1']

sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
        'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
        'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b']
script = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/GLM/cache_glm.py'
python_path = '/auto/users/hellerc/anaconda3/envs/crh_nems/bin/python'

nd.enqueue_models(celllist=sites,
                  batch=batch,
                  modellist=modelnames,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  reserve_gb=4)
