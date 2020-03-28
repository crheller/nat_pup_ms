"""
Copy of 'cache_dprime' - This copy, for a given site, for latent variable models,
will save the dprime results for all hyperparam values. Save structure then looks like:
    results_xforms/site/a0:0/dprime_pr_lvr.csv'
This particular file (cache_dprime_xforms2.py) was made for the following modelstring:
    fast LV + pupil model. Compute dprime for all (20) hyperparam values of the
    fast LV, then save to results_xforms2
"""

import nems.db as nd
import nems.xforms as xforms
import nems_lbhb.baphy as nb
from nems_lbhb.preprocessing import mask_high_repetion_stims, create_pupil_mask
from nems.recording import Recording
import sys
sys.path.append('/auto/users/hellerc/code/projects/nat_pupil_ms/')
import load_results as ld
sys.path.append('/auto/users/hellerc/code/charlieTools/')
import charlieTools.simulate_data as sim
import charlieTools.discrimination_tools as di
import charlieTools.preprocessing as preproc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import nems
import copy
import logging

log = logging.getLogger(__name__)

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick

else:
    queueid = 0

if queueid:
    log.info("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

# first system argument is the cellid
site = sys.argv[1]  # very first (index 0) is the script to be run
# second systems argument is the batch
batch = sys.argv[2]
# third system argument in the modelname
modelname = sys.argv[3]

chan_nums = None
savesite = site
if '.e1:64' in site:
    site = site.split('.e')[0]
    chan_nums = [str(x) for x in np.arange(0, 65)]
elif '.e65:128' in site:
    site = site.split('.e')[0]
    chan_nums = [str(x) for x in np.arange(65, 129)]

path = '/auto/users/hellerc/results/nat_pupil_ms/dprime/'

# parse model options
all_data = '_all' in modelname
big_pupil = '_bp' in modelname
small_pupil = '_sp' in modelname
simulate = '_sim' in modelname
pupil_regressed = '_pr' in modelname
lv_regress = '_lvr' in modelname
regression_method1 = '_rm1' in modelname
regression_method2 = '_rm2' in modelname
single_decoder = '_sia' in modelname # sia - state independent axis
balanced = '_bal' in modelname # use xforms model fit on pupil balanced epochs 
                               # (otherwise, do the correction on all the data, then mask balanced epochs)
permute_pupil = '_permutation' in modelname

log.info('Analyzing site: {0} with options: \n \
                all data: {1} \n \
                big pupil: {2} \n \
                small pupil: {3} \n \
                simulate: {4} \n \
                pupil regressed: {5} \n \
                lv regressed: {6} \n \
                balanced xforms model: {7} \n \
                permuting pupil: {8}'.format(savesite, all_data, big_pupil,
                small_pupil, simulate, pupil_regressed, lv_regress, balanced, permute_pupil))
log.info("Saving results to: {}".format(path))


batch = int(batch)
fs = 4

# new (hopefully FINAL) list of xforms models for dprime:
xforms_models = [
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,5-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,5-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,5-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,5-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,5-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,5-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,5-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,5-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,5-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,5-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,5-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,5-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp1-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp1-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3'
]

# if permute pupil, set random number generator and set range
if permute_pupil:
    np.random.seed(int(modelname.split('_permutation')[-1]))

'''
# additional filtering models
xforms_models = [
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,01-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,01-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,01-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,01-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,01-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,01-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,1',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,01-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,01-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,01-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,2',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,01-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred.hp0,01-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3',
    'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred.hp0,01-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3'
]
'''

if batch == 294:
    xforms_models = [x.replace('fs4.pup', 'fs4.pup.voc') for x in xforms_models]

if not balanced:
    xforms_models = [x.replace('-pbal', '') for x in xforms_models]

if lv_regress | pupil_regressed & regression_method2:
    # regression with model pred. Need to test with all models.
    pass
elif lv_regress & regression_method1:
    # regression with brute force, but LV depends on model. Need to test with all models.
    pass
elif pupil_regressed & regression_method1:
    # regression with brute force. Pupil timeseries is the same no matter which model you load.
    xforms_models = [xforms_models[0]]
else:
    # No regression. Load any model.
    xforms_models = [xforms_models[0]]
    
for count, xforms_modelname in enumerate(xforms_models):
    log.info("Load xforms model {0} / {1} models".format(count+1, len(xforms_models)))
    log.info("Load recording from xforms model {}".format(xforms_modelname))
    if chan_nums is not None:
        cellid = [[c for c in nd.get_batch_cells(batch).cellid if (site in c) & (c.split('-')[1] in chan_nums)][0]]
    else:
        cellid = [[c for c in nd.get_batch_cells(batch).cellid if site in c][0]]
    mp = nd.get_results_file(batch, [xforms_modelname], cellid).modelpath[0]
    xfspec, ctx = xforms.load_analysis(mp)
    # apply hrc (and balanced, if exists) and evoked masks from xforms fit
    rec = ctx['val'].apply_mask(reset_epochs=True)
    
    rec = rec.create_mask(True)

    # filtering / pupil regression must always go first!
    if pupil_regressed & lv_regress:

        if regression_method1:
            log.info('Regress first and second order pupil using brute force method')
            if rec['lv']._data.shape[0] > 1:
                rec['lv'] = rec['pupil']._modified_copy(rec['lv']._data[1:,:])
            rec = preproc.regress_state(rec, state_sigs=['pupil', 'lv'], regress=['pupil', 'lv'])
        elif regression_method2:
            log.info('Regress first and second order pupil by subtracting model pred')
            mod_data = rec['resp']._data - rec['pred']._data + rec['psth_sp']._data
            rec['resp'] = rec['resp']._modified_copy(mod_data)
        else:
            raise ValueError("No regression method specified!")

    elif pupil_regressed:

        if regression_method1:
            log.info('Regress first order pupil using brute force method')
            rec = preproc.regress_state(rec, state_sigs=['pupil'], regress=['pupil'])
        elif regression_method2:
            log.info('Regress first order pupil by subtracting model pred using pupil pred alone')
            # set the LV to 0 by setting encoding weights to 0 and compute prediction
            ms = ctx['modelspec']
            ms.phi[1]['e'] = np.zeros(ms.phi[1]['e'].shape)
            pred = ms.evaluate(rec)
            mod_data = rec['resp']._data - pred['pred']._data + rec['psth_sp']._data
            rec['resp'] = rec['resp']._modified_copy(mod_data)
        else:
            raise ValueError("No regression method specified!")

    #if permute_pupil:
    #    # randomize pupil
    #    pupil = rec['pupil']._data.copy()
    #    np.random.shuffle(pupil.T)
    #    rec['pupil'] = rec['pupil']._modified_copy(pupil)
    #    rec_bp = create_pupil_mask(rec, **{'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True})
    #    rec_sp = create_pupil_mask(rec, **{'state': 'small', 'epoch': ['REFERENCE'], 'collapse': True})

    #else:
    # USE XFORMS PUPIL MASK
    rec_bp = rec.copy()
    rec_bp['mask'] = rec_bp['p_mask']
    rec_sp = rec.copy()
    rec_sp['mask'] = rec_sp['p_mask']._modified_copy(~rec_sp['p_mask']._data)

    rec_bp = rec_bp.apply_mask(reset_epochs=True)
    rec_sp = rec_sp.apply_mask(reset_epochs=True)

    # Determine if need to mask pupil balanced epochs only, or if the 
    # xforms model already did this
    if not balanced:
        if batch == 289:
            balanced_eps = preproc.get_pupil_balanced_epochs(rec, rec_sp, rec_bp)
        elif batch == 294:
            # no need to balanced for batch 294, many reps of each
            balanced_eps = np.unique([s for s in rec.epochs.name if 'STIM' in s]).tolist()
        else:
            raise ValueError("unknown batch")


        if len(balanced_eps)==0:
            log.info("No balanced epochs to extract, quitting")
            sys.exit()

        log.info("Extracting spike count dictionaries for big pupil, \
            small pupil, and all balanced pupil trials.")
        real_dict_all = rec['resp'].extract_epochs(balanced_eps)
        real_dict_small = rec_sp['resp'].extract_epochs(balanced_eps)
        real_dict_big = rec_bp['resp'].extract_epochs(balanced_eps)
    else:
        # xforms model already has balanced epochs
        log.info("Extracting spike count dictionaries for big pupil, \
            small pupil, and all balanced pupil trials.")
        epochs = np.unique([s for s in rec.epochs.name if 'STIM' in s]).tolist()
        real_dict_all = rec['resp'].extract_epochs(epochs)
        real_dict_small = rec_sp['resp'].extract_epochs(epochs)
        real_dict_big = rec_bp['resp'].extract_epochs(epochs)

    # now, if permute, randomize the id of each trial as large / small pupil
    # but preserve the stimulus identity. Do this by randomly choosing 
    # half of the real_dict_all trials and assigning them to large, and the
    # other half assign to small.
    if permute_pupil:
        for k in real_dict_all.keys():
            data = real_dict_all[k]
            all_trials = np.arange(0, data.shape[0])
            large_trials = np.random.choice(all_trials, int(len(all_trials) / 2), replace=False)
            small_trials = np.array(list(set(all_trials) - set(large_trials)))
            real_dict_small[k] = data[small_trials, :, :]
            real_dict_big[k] = data[large_trials, :, :]


    # always use the same data (all trials raw/filtered/regressed) for computing decoding axis
    # doesn't really matter if it's made before/after simulation bc mean doesn't change
    if single_decoder:
        decoding_dict = copy.deepcopy(real_dict_all)
    else:
        # force the data used for computing decoding axis to be the same as the 
        # data to be decoded.
        if big_pupil:
            decoding_dict = copy.deepcopy(real_dict_big)
        elif small_pupil:
            decoding_dict = copy.deepcopy(real_dict_small)
        else:
            # all data case
            decoding_dict = copy.deepcopy(real_dict_all)

    # preprocess/simulate based on options for dprime calculation
    if simulate & big_pupil:

        if '_sim12' in modelname:
            log.info('Keep first and second order stats')
            real_dict_big = sim.generate_simulated_trials(real_dict_big, 
                                        r2=None,
                                        keep_stats=[1, 2], N=5000)

        elif '_sim1' in modelname:

            if '_uMatch' in modelname:
                log.info('Keep first order stats, impose second order stats over all the data')
                real_dict_big = sim.generate_simulated_trials(real_dict_big, 
                                            r2=real_dict_all, 
                                            keep_stats=[1], N=5000)
            else:
                log.info('Keep first order stats, impose second order stats from small pupil')
                real_dict_big = sim.generate_simulated_trials(real_dict_big, 
                                            r2=real_dict_small, 
                                            keep_stats=[1], N=5000)

        elif '_sim2' in modelname:
            
            if '_uMatch' in modelname:
                log.info('Keep second order stats, impose first order stats over all the data')
                real_dict_big = sim.generate_simulated_trials(real_dict_big, 
                                            r2=real_dict_all, 
                                            keep_stats=[2], N=5000)
            else:
                log.info('Keep second order stats, impose first order stats from small pupil')
                real_dict_big = sim.generate_simulated_trials(real_dict_big, 
                                            r2=real_dict_small, 
                                            keep_stats=[2], N=5000)


        if ('_simDec' in modelname) & single_decoder:
            log.info("using the simulated data to define the decoding axis")
            # use the simulated data as the decoding axis
            decoding_dict = copy.deepcopy(real_dict_big)


    elif simulate & small_pupil:

        if '_sim12' in modelname:
            log.info('Keep first and second order stats')
            real_dict_small = sim.generate_simulated_trials(real_dict_small, 
                                        r2=None,
                                        keep_stats=[1, 2], N=5000)

        elif '_sim1' in modelname:

            if '_uMatch' in modelname:
                log.info('Keep first order stats, impose second order stats over all the data')
                real_dict_small = sim.generate_simulated_trials(real_dict_small, 
                                            r2=real_dict_all, 
                                            keep_stats=[1], N=5000)
            else:
                log.info('Keep first order stats, impose second order stats from big pupil')
                real_dict_small = sim.generate_simulated_trials(real_dict_small, 
                                            r2=real_dict_big, 
                                            keep_stats=[1], N=5000)

        elif '_sim2' in modelname:
            
            if '_uMatch' in modelname:
                log.info('Keep second order stats, impose first order stats over all the data')
                real_dict_small = sim.generate_simulated_trials(real_dict_small, 
                                            r2=real_dict_all, 
                                            keep_stats=[2], N=5000)
            else:
                log.info('Keep second order stats, impose first order stats from big pupil')
                real_dict_small = sim.generate_simulated_trials(real_dict_small, 
                                            r2=real_dict_big, 
                                            keep_stats=[2], N=5000)

        if ('_simDec' in modelname) & single_decoder:
            log.info("using the simulated data to define the decoding axis")
            # use the simulated data as the decoding axis
            decoding_dict = copy.deepcopy(real_dict_small)

    # now, compute dprime for the specified conditions
    log.info("perform dprime calculation...")
    if big_pupil:
        try:
            spont_bins = rec_bp['resp'].extract_epoch('PreStimSilence').shape[-1]
        except:
            log.info('No spont bins, xforms model fit on evoked only data')
            spont_bins = None

        dp = di.compute_dprime_from_dicts(real_dict_big, 
                                        decoding_dict, 
                                        norm=True,
                                        LDA=False, 
                                        spont_bins=spont_bins)

    elif small_pupil:
        try:
            spont_bins = rec_sp['resp'].extract_epoch('PreStimSilence').shape[-1]
        except:
            log.info('No spont bins, xforms model fit on evoked only data')
            spont_bins = None

        dp, d = di.compute_dprime_from_dicts(real_dict_small, 
                                        decoding_dict, 
                                        norm=True,
                                        LDA=False, 
                                        spont_bins=spont_bins, verbose=True)

    elif all_data:
        try:
            spont_bins = rec['resp'].extract_epoch('PreStimSilence').shape[-1]
        except:
            log.info('No spont bins,  xforms model fit on evoked only data')
            spont_bins = None

        dp = di.compute_dprime_from_dicts(real_dict_all, 
                                        decoding_dict,
                                        norm=True,
                                        LDA=False, 
                                        spont_bins=spont_bins)

    else:
        raise ValueError("No dataset specified for dprime calculation")

    log.info("save dprime data frame for model: {0}, site: {1}".format(modelname, savesite))
    n = dp.shape[0]
    sites = [savesite] * n
    dp['site'] = sites
    a = xforms_modelname
    spath = path+savesite+'/'
    fpath = path+savesite+'/'+a+'/'

    if os.path.isdir(spath):
        pass
    else:
        os.mkdir(spath, mode=0o777)

    if os.path.isdir(fpath):
        pass
    else:
        os.mkdir(fpath, mode=0o777)

    dp.to_csv(fpath+modelname+'.csv')

if queueid:
    nd.update_job_complete(queueid)
