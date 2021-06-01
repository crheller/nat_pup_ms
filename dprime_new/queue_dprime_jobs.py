import nems.db as nd
import numpy as np
from global_settings import CPN_SITES

batch = 331
njack = 10
force_rerun = True
subset_289 = True  # only high rep sites (so that we can do cross validation)
subset_323 = False # only high rep sites (for cross val)
no_crossval = False  # for no cross validation (on the larger 289 set )
lvmodels = False    # run for the simulated, model results from lv xforms models
pca = False
pc_keys = ['pca-3-psth-whiten', 'pca-4-psth-whiten', 'pca-5-psth-whiten']
pc_keys = ['pca-4-psth-whiten']
zscore = True

temp_subset = False # for exculding subset of models/sites for faster run time on jobs

nc_lv = True        # beta defined using nc LV method
fix_tdr2 = True     # force tdr2 axis to be defined based on first PC of POOLED noise data. Not on a per stimulus basis.
ddr2_method = 'fa'
exclude_lowFR = False
thresh = 1 # minimum mean FR across all conditions
sim_in_tdr = True   # for sim1, sim2, and sim12 models, do the simulation IN the TDR space.
loocv = False         # leave-one-out cross validation
n_additional_noise_dims = 0 # how many additional TDR dims? 0 is the default, standard TDR world. additional dims are controls
NOSIM = True   # If true, don't run simulations

# each of these corresponds to an xforms LV model. If modelname = model-LV-modelname, 
# "resp" will be loaded from the pred of "modelname", then decoding will be performed.
lvmodelnames = {
    "psth.fs4.pup-loadpred.pc4-st.pup0.pvp0-plgsm.e-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_pcnorm.t6",
    "psth.fs4.pup-loadpred.pc4-st.pup.pvp0-plgsm.e-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_pcnorm.t6",
    "psth.fs4.pup-loadpred.pc4-st.pup.pvp-plgsm.e-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_pcnorm.t6"
}
lvmodelnames = {
    "psth.fs4.pup-loadpred.pc4-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_pcnorm.t6",
    "psth.fs4.pup-loadpred.pc4-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_pcnorm.t6",
    "psth.fs4.pup-loadpred.pc4-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_pcnorm.t6"
}
lvmodelnames = {
    "psth.fs4.pup-loadpred.pc4-st.pup0.pvp0-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6",
    "psth.fs4.pup-loadpred.pc4-st.pup.pvp0-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6",
    "psth.fs4.pup-loadpred.pc4-st.pup.pvp-plgsm.e5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6",
    "psth.fs4.pup-loadpred.pc4-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6",
    "psth.fs4.pup-loadpred.pc4-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6",
    "psth.fs4.pup-loadpred.pc4-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6"
}

lvmodelnames = {
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5.ss3",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5.ss3",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.e10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t5.ss3"
}

lvmodelnames = {
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss1",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss2",
    "psth.fs4.pup-loadpred-st.pup.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3",
    "psth.fs4.pup-loadpred-st.pup0.pvp0-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.SxR.d-inoise.2xR_ccnorm.t6.ss3"
}

# indep noise only model
lvmodelnames = {
    "psth.fs4.pup-loadpred-st.pup0.pvp-plgsm.eg5.sp-lvnoise.r8-aev_lvnorm.2xR.d-inoise.3xR_ccnorm.t5.ss3",
    "psth.fs4.pup-loadpred-st.pup0.pvp-plgsm.eg10.sp-lvnoise.r8-aev_lvnorm.2xR.d-inoise.3xR_ccnorm.t5.ss3"
}

if no_crossval & loocv:
    raise ValueError("loocv implies no_crossval (eev). Only set one or the other true")

if batch == 289:
    sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
            'DRX006b.e1:64', 'DRX006b.e65:128',
            'DRX007a.e1:64', 'DRX007a.e65:128',
            'DRX008b.e1:64', 'DRX008b.e65:128', 'CRD016d', 'CRD017c']
    if subset_289:
        # list of sites with > 10 reps of each stimulus
        sites = ['TAR010c', 'TAR017b', 
                'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
                'DRX007a.e1:64', 'DRX007a.e65:128', 
                'DRX008b.e1:64', 'DRX008b.e65:128', 
                'CRD016d', 'CRD017c']
            
elif batch == 294:
    sites = ['BOL005c', 'BOL006b']

elif batch == 331:
    sites = CPN_SITES

elif batch == 323:
    if subset_323:
        sites = ['ARM018a', 'ARM019a', 'ARM021b', 'ARM022b']
    else:
        sites = ['AMT028b', 'AMT029a', 'AMT031a', 'AMT032a']

#modellist = ['dprime_jk10']
modellist = [f'dprime_jk{njack}_zscore', f'dprime_pr_jk{njack}_zscore',
            f'dprime_sim1_jk{njack}_zscore', f'dprime_sim12_jk{njack}_zscore',
            f'dprime_sim1_pr_jk{njack}_zscore', f'dprime_sim12_pr_jk{njack}_zscore',
            f'dprime_pr_rm2_jk{njack}_zscore', 
            f'dprime_sim1_pr_rm2_jk{njack}_zscore', f'dprime_sim12_pr_rm2_jk{njack}_zscore']

# NOTE: as of 06.04.2020: tried regressing out only baseline or only gain (prd / prg models). Didn't see much of 
# a difference. Still an option though. May want to look into a bug at some point.
# NOTE: as of 08.23.2020: removed sim2 models from queue list. Decided to just use first order and first + second order

if pca:
    new = []
    for pc_key in pc_keys:
        new.extend([m.replace('dprime_', f'dprime_{pc_key}_') for m in modellist])
    modellist = new

if no_crossval:
    modellist = [m.replace('_jk10', '_jk1_eev') for m in modellist]

if loocv:
    modellist = [m.replace('_jk10', '_jk1_loocv') for m in modellist]

if sim_in_tdr:
    modellist = [m.replace('_sim', '_simInTDR_sim') for m in modellist]

if nc_lv:
    modellist = [m.replace('zscore', 'zscore_nclvz') for m in modellist]

if fix_tdr2:
    if (ddr2_method=='pca') | (ddr2_method is None):
        modellist = [m+'_fixtdr2' for m in modellist]
    else:
        modellist = [m+f'_fixtdr2-{ddr2_method}' for m in modellist]

if temp_subset:
    sites = [s for s in sites if 'CRD' in s]
    #modellist = [m for m in modellist if ('_sim' in m)]

if n_additional_noise_dims > 0:
    modellist = [m+'_noiseDim{0}'.format(n_additional_noise_dims) for m in modellist]

if NOSIM:
    modellist = [m for m in modellist if ('_sim1' not in m) & ('_sim2' not in m) & ('_sim12' not in m)]

if lvmodels:
    # don't do the pupil regression models for this, doesn't make sense
    modellist = [m for m in modellist if '_pr_' not in m]
    modellist = np.concatenate([[m+f'_model-LV-{lvstr}' for lvstr in lvmodelnames] for m in modellist]).tolist()
    if batch == 331:
        modellist = [m.replace('loadpred', 'loadpred.cpn') for m in modellist]

if zscore == False:
    modellist = [m.replace('_zscore', '') for m in modellist]

if exclude_lowFR:
    modellist = [m+f'_rmlowFR-{thresh}' for m in modellist]

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
