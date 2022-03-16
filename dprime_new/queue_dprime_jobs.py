from logging import shutdown
import nems.db as nd
import numpy as np
from global_settings import CPN_SITES, HIGHR_SITES

mvm_masks = [None, (25, 1), (25, 2)] # (threshold (in sd*100) and binsize (in sec)) -- for raw data analsysi
mvm_masks = [None] # for NAT data, would need to reanalyze pupil
noise_dims = [-1, 0, 1, 2, 3, 4, 5, 6]

thirds = False # create 3 additional jobs that split pupil up into thirds -- co-opt the "bp" mask for this.
               # sort of weird, but idea is that then nothing changes (calculation of decoding axis etc.),
               # we just project a different set of data into the "thing" classified as "big pupil". Kludgy,
               # but should work fine.
allPup = False  # project all pupil data (full mask, not kacnkifed) into the est decoding space to prevent low rep count stuff from happening
               # alternative is to set this to false, and all stim where <5 reps of each in each state won't be used.
batch = 294 # note, slowly replacing 29.10.2021, replacing batch 289 with 322 (this is where array NAT data is stored. 289 should have all)
njack = 10
force_rerun = True
subset_289 = True  # only high rep sites (so that we can do cross validation)
subset_323 = False # only high rep sites (for cross val)
subset_331 = False # specialized subset of 331 data (e.g. only run for a subset of sites that are new data)
no_crossval = False  # ******** for no cross validation (on the larger 289 set ) and/or debugging purposes
pca = False
pc_keys = ['pca-3-psth-whiten', 'pca-4-psth-whiten', 'pca-5-psth-whiten']
pc_keys = ['pca-4-psth-whiten']
zscore = True     # ********
shuffle = False
temp_subset = False # for exculding subset of models/sites for faster time on jobs
nc_lv = False        # beta defined using nc LV method (if False, don't bother loading betas -- 09.08.2021, I think this is what we want. nclv isn't super relevant anymore)
fix_tdr2 = True     # force tdr2 axis to be defined based on first PC of POOLED noise data. Not on a per stimulus basis.
ddr2_method = 'fa' #'nclv'  # None, 'fa', 'nclv'
exclude_lowFR = False
thresh = 1 # minimum mean FR across all conditions
sim_in_tdr = True   # for sim1, sim2, and sim12 models, do the simulation IN the TDR space.
loocv = False         # leave-one-out cross validation
NOSIM = True   # If true, don't run simulations

lvmodels = False   # run for the simulated, model results from lv xforms models
lvmodelset = "2022.02.06" # options were getting confusing, just organize by date
faModel = True # load results of factor analysis models for the site, then generate data based on these low-D representations 
fa_ind_var = False # allow ind. variance to change between stimuli (makes sense if gain). fa model only models shared noise between stim by default
fa_ind_var_only = True  # only allow ind. variance to change. FA shared factors are fixed between states.

use_old_cpn = False

for movement_mask in mvm_masks:
    for n_additional_noise_dims in noise_dims:
        if lvmodels:
            # define list of lv models to fit 
            if lvmodelset=="2022.02.06":
                lvmodelnames = [
                    'psth.fs4.pup-ld-st.pup+r2+s1,2-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.2xR.x2,3-spred-lvnorm.4xR.so.x1-inoise.4xR.x1,3_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss1.rb5',
                    'psth.fs4.pup-ld-st.pup+r2+s1,2-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.2xR.x2,3-spred-lvnorm.4xR.so.x1-inoise.4xR.x2,3_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss1.rb5',
                    'psth.fs4.pup-ld-st.pup+r2+s1,2-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.2xR.x2,3-spred-lvnorm.4xR.so.x3-inoise.4xR.x2,3_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss1.rb5',
                    'psth.fs4.pup-ld-st.pup+r2-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.2xR.x2,3-spred-lvnorm.4xR.so.x3-inoise.4xR.x2,3_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss1.rb5'
                ]
                ss2 = [mn.replace(".ss1", ".ss2") for mn in lvmodelnames]
                ss3 = [mn.replace(".ss1", ".ss3") for mn in lvmodelnames]
                lvmodelnames = lvmodelnames + ss2 + ss3

            else:
                raise DeprecationWarning("Thise was for old models. Do this by 'hand' now with dates (see above)")

        if no_crossval & loocv:
            raise ValueError("loocv implies no_crossval (eev). Only set one or the other true")

        if batch == 289:
            sites = HIGHR_SITES
            sites = [s for s in sites if s not in ['BOL005c', 'BOL006b']]
        
        elif batch == 322:
            sites = HIGHR_SITES
            sites = [s for s in sites if s not in ['BOL005c', 'BOL006b']]
                    
        elif batch == 294:
            sites = ['BOL005c', 'BOL006b']

        elif batch == 331:
            sites = CPN_SITES
            if subset_331:
                input("NOTE -- using only a subset of batch 331 sites. Okay?")
                sites = [s for s in sites if (s!='TNC016a') & (s!='TNC013a')]

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

        if allPup:
            modellist = [m+'_allPup' for m in modellist]
        
        if thirds:
            modellist13 = [m+'_png13' for m in modellist]
            modellist23 = [m+'_png23' for m in modellist]
            modellist33 = [m+'_png33' for m in modellist]
            bmodellist13 = [m+'_pbg13' for m in modellist]
            bmodellist23 = [m+'_pbg23' for m in modellist]
            bmodellist33 = [m+'_pbg33' for m in modellist]
            modellist = modellist + modellist13 + modellist23 + modellist33 + bmodellist13 + bmodellist23 + bmodellist33

        # NOTE: as of 06.04.2020: tried regressing out only baseline or only gain (prd / prg models). Didn't see much of 
        # a difference. Still an option though. May want to look into a bug at some point.
        # NOTE: as of 08.23.2020: removed sim2 models from queue list. Decided to just use first order and first + second order

        if shuffle:
            modellist = [m.replace('dprime_', 'dprime_shuffle_') for m in modellist]

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
            modellist = [m+'_noiseDim-{0}'.format(n_additional_noise_dims) for m in modellist]

        if n_additional_noise_dims < 0:
            modellist = [m+'_noiseDim{0}'.format('-dU') for m in modellist]

        if NOSIM:
            modellist = [m for m in modellist if ('_sim1' not in m) & ('_sim2' not in m) & ('_sim12' not in m)]

        if lvmodels:
            # don't do the pupil regression models for this, doesn't make sense
            modellist = [m for m in modellist if '_pr_' not in m]
            modellist = np.concatenate([[m+f'_model-LV-{lvstr}' for lvstr in lvmodelnames] for m in modellist]).tolist()
            if batch != 331:
                modellist = [m.replace('-epcpn', "") for m in modellist]

        if zscore == False:
            modellist = [m.replace('_zscore', '') for m in modellist]

        if exclude_lowFR:
            modellist = [m+f'_rmlowFR-{thresh}' for m in modellist]

        if movement_mask is not None:
            modellist = [m.replace('dprime_', f'dprime_mvm-{movement_mask[0]}-{movement_mask[1]}_') for m in modellist]

        if use_old_cpn:
            modellist = [m.replace('dprime_', f'dprime_oldCPN_') for m in modellist]
        
        if faModel:
            if fa_ind_var:
                modellist = [m.replace("dprime_", "dprime_faModel.ind_") for m in modellist]
            elif fa_ind_var_only:
                modellist = [m.replace("dprime_", "dprime_faModel.ind-null_") for m in modellist]
            else:
                modellist = [m.replace("dprime_", "dprime_faModel_") for m in modellist]

        modellist = [m for m in modellist if '_pr' not in m]

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
