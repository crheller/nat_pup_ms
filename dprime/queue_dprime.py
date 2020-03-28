#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:28:10 2018

@author: hellerc
"""
import nems.db as nd
import numpy as np

batch = 294
update = False
force_rerun = True
permutation = False
balanced = False # use xforms fit over all evoked data, then balance epochs inside the dprime code
#balanced = True  # use xforms fit with balanced pupil epochs (if False, dprime code will still balance epochs)

correction_method = 1 # do regression with brute force method (1) or by subtracting xforms prediction (2)

if permutation:
    bp_modelname = 'dprime_bp_sia_permutation'
    sp_modelname = 'dprime_sp_sia_permutation'
    bp_modelnames = [bp_modelname+str(p) for p in np.arange(0, 200)]
    sp_modelnames = [sp_modelname+str(p) for p in np.arange(0, 200)]
    modelnames = bp_modelnames + sp_modelnames
else:
    modelnames = ['dprime_all',
                'dprime_bp',
                'dprime_sp',
                'dprime_all_pr',
                'dprime_all_pr_lvr',
                'dprime_bp_pr',
                'dprime_sp_pr',
                'dprime_bp_pr_lvr',
                'dprime_sp_pr_lvr',
                'dprime_bp_sim1',
                'dprime_bp_sim2',
                'dprime_sp_sim1',
                'dprime_sp_sim2',
                'dprime_bp_sim12',
                'dprime_sp_sim12',
                'dprime_bp_pr_sim1',
                'dprime_bp_pr_sim2',
                'dprime_sp_pr_sim1',
                'dprime_sp_pr_sim2',
                'dprime_bp_pr_sim12',
                'dprime_sp_pr_sim12',
                'dprime_bp_pr_lvr_sim1',
                'dprime_bp_pr_lvr_sim2',
                'dprime_sp_pr_lvr_sim1',
                'dprime_sp_pr_lvr_sim2',
                'dprime_bp_pr_lvr_sim12',
                'dprime_sp_pr_lvr_sim12']

    if balanced:
        modelnames = [m+'_bal' for m in modelnames]

    # state independent decoding axis (sia)
    modelnames = [m+'_sia'  if 'all' not in m else m for m in modelnames]

    # set the regression correction method
    modelnames = [m+'_rm{}'.format(correction_method) if ('_pr' in m) | ('_lvr' in m) else m for m in modelnames]

    # uMatch means that for simulations, simulate relative to the entire data set
    # for example, sp_sim1_uMatch means keep first order pupil stats, but impose 
    # the second order stats measured over all the data. Without uMatch, the default is 
    # to use the second order stats from the opposite condition (in this case, big pupil)
    modelnames = [m+'_uMatch'  if 'sim' in m else m for m in modelnames]  

    # only keep regression models
    #modelnames = [m for m in modelnames if ('_pr' in m) | ('_lvr' in m)]

if (batch == 289) & update:
    #update batch 289 03/10 - needed to (re) run some models.
    sites = ['DRX006b.e1:64', 'DRX006b.e65:128',
             'DRX007a.e1:64', 'DRX007a.e65:128',
             'DRX008b.e1:64', 'DRX008b.e65:128']

elif batch == 289:
    sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
            'DRX006b.e1:64', 'DRX006b.e65:128',
            'DRX007a.e1:64', 'DRX007a.e65:128',
            'DRX008b.e1:64', 'DRX008b.e65:128']
            
elif batch == 294:
    sites = ['BOL005c', 'BOL006b']
     
script = '/auto/users/hellerc/code/projects/nat_pupil_ms/dprime2/cache_dprime.py'
python_path = '/auto/users/hellerc/anaconda3/envs/crh_nems/bin/python'

nd.enqueue_models(celllist=sites,
                  batch=batch,
                  modellist=modelnames,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  reserve_gb=4)