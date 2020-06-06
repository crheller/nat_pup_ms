import nems.db as nd
import numpy as np

batch = 294
force_rerun = True
subset_289 = True  # only high rep sites (so that we can do cross validation)
temp_subset = True # for exculding subset of models for faster run time on jobs
nc_lv = True       # beta defined using nc LV method
fix_tdr2 = True    # force tdr2 axis to be defined based on first PC of POOLED noise data. Not on a per stimulus basis.
sim_in_tdr = True  # for sim1, sim2, and sim12 models, do the simulation IN the TDR space.

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
            'dprime_sim1_pr_rm2_jk10_zscore', 'dprime_sim2_pr_rm2_jk10_zscore',  'dprime_sim12_pr_rm2_jk10_zscore']

# NOTE: as of 06.04.2020: tried regressing out only baseline or only gain (prd / prg models). Didn't see much of 
# a difference. Still an option though. May want to look into a bug at some point.

if sim_in_tdr:
    modellist = [m.replace('_sim', '_simInTDR_sim') for m in modellist]

if nc_lv:
    modellist = [m.replace('zscore', 'zscore_nclvz') for m in modellist]

if fix_tdr2:
    modellist = [m+'_fixtdr2' for m in modellist]

if temp_subset:
    modellist = [m for m in modellist if ('_sim' in m)]

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