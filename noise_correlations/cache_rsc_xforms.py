
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
import charlieTools.noise_correlations as nc
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
batch = int(sys.argv[2])
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

pupil_regressed = 'pr' in modelname
lv_regress = 'lvr' in modelname
balanced = 'bal' in modelname
regression_method1 = 'rm1' in modelname
regression_method2 = 'rm2' in modelname
filt = 'fft' in modelname

keys = modelname.split('_')
for k in keys:
    if 'fft' in k:
        low_c = np.float(k.split('-')[0][3:])
        high_c = np.float(k.split('-')[1])

path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations/'

log.info('Computing noise correlations for site: {0} with options: \n \
            regress pupil: {1} \n \
            regress lv: {2} \n \
            balanced pupil epochs: {3}'.format(savesite, pupil_regressed, lv_regress, balanced))

log.info("Saving results to: {}".format(path))

batch = int(batch)

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

for xforms_modelname in xforms_models:
    log.info("Load recording from xforms model {}".format(xforms_modelname))
    if chan_nums is not None:
        cellid = [[c for c in nd.get_batch_cells(batch).cellid if (site in c) & (c.split('-')[1] in chan_nums)][0]]
    else:
        cellid = [[c for c in nd.get_batch_cells(batch).cellid if site in c][0]]
        
    mp = nd.get_results_file(batch, [xforms_modelname], cellid).modelpath[0]
    xfspec, ctx = xforms.load_analysis(mp)
    # apply hrc (and balanced, if exists) and evoked masks from xforms fit
    rec = ctx['val'].apply_mask(reset_epochs=True)
    
    # only necessary if using correction method 1
    if lv_regress:
        if rec['lv']._data.shape[0] > 1:
            rec['lv'] = rec['pupil']._modified_copy(rec['lv']._data[1:,:])
    
    rec = rec.create_mask(True)

    # filtering / pupil regression must always go first!
    if pupil_regressed & lv_regress:

        if regression_method1:
            log.info('Regress first and second order pupil using brute force method')
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

    if filt:
        log.info("Band-pass filter spike counts between {0} and {1} Hz".format(low_c, high_c))
        rec = preproc.bandpass_filter_resp(rec, low_c, high_c)

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

    # compute noise correlations and save results
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
    df['site'] = savesite

    log.info("save noise corr results for model: {0}, site: {1}".format(modelname, savesite))
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

    df.to_csv(fpath+modelname+'.csv')


if queueid:
    nd.update_job_complete(queueid)
