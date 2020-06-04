import nems.db as nd
import numpy as np

batch = 289
force_rerun = False
subset_289 = True
temp_subset = True # for exculding subset of models for faster run time on jobs
nc_lv = True
fix_tdr2 = True

if batch == 289:
    sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
            'DRX006b.e1:64', 'DRX006b.e65:128',
            'DRX007a.e1:64', 'DRX007a.e65:128',
            'DRX008b.e1:64', 'DRX008b.e65:128']
    if subset_289:
        # list of sites with > 10 reps of each stimulus
        sites = ['TAR010c', 'TAR017b', 
                'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
                'DRX007a.e1:64', 'DRX007a.e65:128', 
                'DRX008b.e1:64', 'DRX008b.e65:128']
            
elif batch == 294:
    sites = ['BOL005c', 'BOL006b']

#modellist = ['dprime_jk10']
modellist = ['dprime_jk10_zscore', 'dprime_pr_jk10_zscore',
            'dprime_sim1_jk10_zscore', 'dprime_sim2_jk10_zscore', 'dprime_sim12_jk10_zscore',
            'dprime_sim1_pr_jk10_zscore', 'dprime_sim2_pr_jk10_zscore', 'dprime_sim12_pr_jk10_zscore',
            'dprime_pr_rm2_jk10_zscore', 
            'dprime_sim1_pr_rm2_jk10_zscore', 'dprime_sim2_pr_rm2_jk10_zscore',  'dprime_sim12_pr_rm2_jk10_zscore',
            'dprime_prg_rm2_jk10_zscore', 
            'dprime_sim1_prg_rm2_jk10_zscore', 'dprime_sim2_prg_rm2_jk10_zscore',  'dprime_sim12_prg_rm2_jk10_zscore',
            'dprime_prd_rm2_jk10_zscore', 
            'dprime_sim1_prd_rm2_jk10_zscore', 'dprime_sim2_prd_rm2_jk10_zscore',  'dprime_sim12_prd_rm2_jk10_zscore']

if nc_lv:
    modellist = [m.replace('zscore', 'zscore_nclvz') for m in modellist]

if fix_tdr2:
    modellist = [m+'_fixtdr2' for m in modellist]

if temp_subset:
    modellist = [m for m in modellist if ('prd' in m)]

script = '/auto/users/hellerc/code/projects/nat_pupil_ms/dprime_new/cache_dprime.py'
python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'

nd.enqueue_models(celllist=sites,
                  batch=batch,
                  modellist=modellist,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  reserve_gb=4)