"""
Compute noise correlations for the following conditions:
    all raw data
    all data pupil regressed
    all data lv regressed (not implemented)

    all of the above for big/small pupil conditions
"""

import nems.db as nd
import nems.xforms as xforms
import nems_lbhb.baphy as nb
from nems_lbhb.preprocessing import mask_high_repetion_stims, create_pupil_mask
from nems.recording import Recording
import sys
sys.path.append('/auto/users/hellerc/code/projects/nat_pupil_ms/')
sys.path.append('/auto/users/hellerc/code/projects/nat_pupil_ms/dprime/')
import load_results as ld
sys.path.append('/auto/users/hellerc/code/charlieTools/')
import charlieTools.simulate_data as sim
import charlieTools.preprocessing as preproc
import charlieTools.noise_correlations as nc
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from itertools import combinations
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

pupil_regress = 'pr' in modelname
lv_regress = 'lvr' in modelname
balanced = 'bal' in modelname
regression_method1 = 'rm1' in modelname
regression_method2 = 'rm2' in modelname
filt = 'fft' in modelname

if lv_regress:
    raise DeprecationWarning("No longer doing LV regression. Script need to be updated to handle this. \n"
                                "CRH 05/08/2020")
if balanced:
    raise DeprecationWarning("Removed pupil balancing option (temporarily?) CRH 05/08/2020")

keys = modelname.split('_')
boxcar = False
for k in keys:
    if 'fft' in k:
        low_c = np.float(k.split('-')[0][3:])
        high_c = np.float(k.split('-')[1])
    if 'boxcar' in k:
        boxcar = True
path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations/'

log.info('Computing noise correlations for site: {0} with options: \n \
            regress pupil: {1} \n \
            regress lv: {2} \n \
            balanced pupil epochs: {3}'.format(site, pupil_regress, lv_regress, balanced))

log.info("Saving results to: {}".format(path))

batch = int(batch)
if filt:
    fs = 100
    xforms_modelname = 'ns.fs100.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'
    #xforms_modelname = 'ns.fs100.pup-ld-st.pup-hrc-psthfr-ev_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.a0:0'
else:
    fs = 4

    if pupil_regress:
        xforms_modelname = 'ns.fs4.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'

    else:
        xforms_modelname = 'ns.fs4.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'

if batch == 294:
    xforms_modelname = xforms_modelname.replace('pup-ld', 'pup.voc-ld')

log.info("Load recording from xforms model {}".format(xforms_modelname))
rec_path = '/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/'

cellid, _ = nb.parse_cellid({'batch': batch, 'cellid': site})
rec = preproc.generate_state_corrected_psth(batch=batch, modelname=xforms_modelname, cellids=cellid, 
                                        siteid=site,
                                        cache_path=rec_path, recache=False)

# filtering / pupil regression must always go first!
if pupil_regress & lv_regress:

    if regression_method1:
        log.info('Regress first and second order pupil using brute force method')
        rec = preproc.regress_state(rec, state_sigs=['pupil', 'lv'], regress=['pupil', 'lv'])
    elif regression_method2:
        log.info('Regress first and second order pupil by subtracting model pred')
        mod_data = rec['resp']._data - rec['pred']._data + rec['psth_sp']._data
        rec['resp'] = rec['resp']._modified_copy(mod_data)
    else:
        raise ValueError("No regression method specified!")

elif pupil_regress:

    if regression_method1:
        log.info('Regress first order pupil using brute force method')
        rec = preproc.regress_state(rec, state_sigs=['pupil'], regress=['pupil'])
    elif regression_method2:
        log.info('Regress first order pupil by subtracting model pred')
        mod_data = rec['resp']._data - rec['psth']._data + rec['psth_sp']._data
        rec['resp'] = rec['resp']._modified_copy(mod_data)
    else:
        raise ValueError("No regression method specified!")

if filt:
    log.info("Band-pass filter spike counts between {0} and {1} Hz".format(low_c, high_c))
    rec = preproc.bandpass_filter_resp(rec, low_c, high_c, boxcar=boxcar)

rec = rec.apply_mask(reset_epochs=True)

log.info("Mask large and small pupil")
# handle using xforms pupil mask
rec_bp = rec.copy()
ops = {'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
rec_bp = create_pupil_mask(rec_bp, **ops)
ops = {'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True}
rec_sp = rec.copy()
rec_sp = create_pupil_mask(rec_sp, **ops)

eps = np.unique([s for s in rec.epochs.name if 'STIM' in s]).tolist()

log.info("Extracting spike count dictionaries for big pupil, \n \
    small pupil, and all pupil trials.")
real_dict_all = rec['resp'].extract_epochs(eps)
real_dict_small = rec_sp['resp'].extract_epochs(eps, mask=rec_sp['mask'])
real_dict_big = rec_bp['resp'].extract_epochs(eps, mask=rec_bp['mask'])

df_all = nc.compute_rsc(real_dict_all, chans=rec['resp'].chans)
df_big = nc.compute_rsc(real_dict_big, chans=rec['resp'].chans)
df_small = nc.compute_rsc(real_dict_small, chans=rec['resp'].chans)


cols = ['all', 'p_all', 'bp', 'p_bp', 'sp', 'p_sp', 'site']
df = pd.DataFrame(columns=cols, index=df_all.index)
df['all'] = df_all['rsc']
df['p_all'] = df_all['pval']
df['bp'] = df_big['rsc']
df['p_bp'] = df_big['pval']
df['sp'] = df_small['rsc']
df['p_sp'] = df_small['pval']
df['site'] = site

log.info("save noise corr results for model: {0}, site: {1}".format(modelname, site))

if os.path.isdir(path+site):
    df.to_csv(path+site+'/'+modelname+'.csv')
else:
    os.mkdir(path+site, 777)
    df.to_csv(path+site+'/'+modelname+'.csv')

if queueid:
    nd.update_job_complete(queueid)
