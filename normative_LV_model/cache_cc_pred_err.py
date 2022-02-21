"""
Load LV models and cache the cc predictions / prediction errors. Use these to evaluate model performance.
"""
from platform import architecture
from global_settings import HIGHR_SITES, CPN_SITES
from nems.xform_helper import load_model_xform
from nems_lbhb.baphy_io import parse_cellid

import numpy as np
import os
import pickle

savedir = "/auto/users/hellerc/results/nat_pupil_ms/normativeModel/modelPreds/"
perstim = True
if perstim:
    filename = "cc_err_perstim.pickle"
else:
    filename = "cc_err.pickle" # how to save results within batch/site/modelname directory
recache = False # typically, this should be true. Only case for false is if cache stopped halfway through and need to rerun for remaining sites
modellist = [
    'psth.fs4.pup-ld-st.pup+r2+s1,2-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.2xR.x2,3-spred-lvnorm.4xR.so.x1-inoise.4xR.x1,3_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss3.rb5',
    'psth.fs4.pup-ld-st.pup+r2+s1,2-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.2xR.x2,3-spred-lvnorm.4xR.so.x1-inoise.4xR.x2,3_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss3.rb5',
    'psth.fs4.pup-ld-st.pup+r2+s1,2-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.2xR.x2,3-spred-lvnorm.4xR.so.x3-inoise.4xR.x2,3_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss3.rb5',
    'psth.fs4.pup-ld-st.pup+r2-epcpn-hrc-psthfr.z-plgsm.er5-aev_stategain.2xR.x2,3-spred-lvnorm.4xR.so.x3-inoise.4xR.x2,3_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss3.rb5'
]
# for saving, come up with shorterened modelnames, more human readable
# note, saved the additive lv models with the wrong name!! (I used indep.d, stupidly)
architecture_spec = "stategain_chanCtrl_ccnorm.md_ss3.rb5"
mn_shortened = [
    f'firstOrderOnly_{architecture_spec}',
    f'indepNoise_{architecture_spec}',
    f'oneLV_{architecture_spec}',
    f'twoLV_{architecture_spec}'
]
# basically a copy of svd function to compare cc matrices
def cc_comp(r, extra_epoch):
    rec = r.apply_mask()
    if type(extra_epoch) is list:
        for i, e in enumerate(extra_epoch):
            if i == 0:
                large_idx=rec['mask_'+e+'_lg'].as_continuous()[0,:].astype(bool)
                small_idx=rec['mask_'+e+'_sm'].as_continuous()[0,:].astype(bool)
            else:
                li = rec['mask_'+e+'_lg'].as_continuous()[0,:].astype(bool)
                si = rec['mask_'+e+'_sm'].as_continuous()[0,:].astype(bool)
                large_idx += li
                small_idx += si
    else:
        large_idx=rec['mask_'+extra_epoch+'_lg'].as_continuous()[0,:].astype(bool)
        small_idx=rec['mask_'+extra_epoch+'_sm'].as_continuous()[0,:].astype(bool)
    print(f"masked {extra_epoch} len from {rec['mask'].as_continuous().sum()} to {large_idx.sum()+small_idx.sum()}")

    input_name = 'pred0'

    pred0 = rec[input_name].as_continuous()
    pred = rec['pred'].as_continuous()
    resp = rec['resp'].as_continuous()
    state = rec['state'].as_continuous()

    large_cc = np.cov(resp[:,large_idx]-pred0[:,large_idx])
    small_cc = np.cov(resp[:,small_idx]-pred0[:,small_idx])
    sm_cc = np.cov(pred[:,small_idx]-pred0[:,small_idx])
    lg_cc = np.cov(pred[:,large_idx]-pred0[:,large_idx])
    all_cc = np.cov(resp[:,large_idx|small_idx]-pred0[:,large_idx|small_idx])
    allp_cc = np.cov(pred[:,large_idx|small_idx]-pred0[:,large_idx|small_idx])

    delta_err_up = np.mean((np.triu(large_cc-small_cc, 1) - np.triu(lg_cc-sm_cc, 1))**2)
    delta_err_diag = np.mean((np.diag(large_cc-small_cc) - np.diag(lg_cc-sm_cc))**2)

    lg_err_up = np.mean((np.triu(large_cc, 1) - np.triu(lg_cc, 1))**2)
    lg_err_diag = np.mean((np.diag(large_cc) - np.diag(lg_cc))**2)

    sm_err_up = np.mean((np.triu(small_cc, 1) - np.triu(sm_cc, 1))**2)
    sm_err_diag = np.mean((np.diag(small_cc) - np.diag(sm_cc))**2)

    avg_err_up = np.mean((np.triu(all_cc, 1) - np.triu(allp_cc, 1))**2)
    avg_err_diag = np.mean((np.diag(all_cc) - np.diag(allp_cc))**2)

    err = {
        "delta_err_up": delta_err_up,
        "delta_err_diag": delta_err_diag,
        "lg_err_up": lg_err_up,
        "lg_err_diag": lg_err_diag,
        "sm_err_up": sm_err_up,
        "sm_err_diag": sm_err_diag,
        "avg_err_up": avg_err_up,
        "avg_err_diag": avg_err_diag
    }

    return err

# ============================== LOOP OVER SITES / BATCHES / MODELNAMES AND SAVE RESULTS =======================================
sites = CPN_SITES + HIGHR_SITES
batches = [331]*len(CPN_SITES) + [322 if s not in ["BOL005c", "BOL006b"] else 294 for s in HIGHR_SITES]
for batch, site in zip(batches, sites):
    print("\n\n")
    print(f"Working on site/batch: {site}/{batch}")
    print("\n\n")
    # make sure save directory exists
    d1 = savedir+f"{site}_{batch}"
    if os.path.isdir(d1):
        pass
    else:
        os.mkdir(d1)
    for (model_key, modelname) in zip(mn_shortened, modellist):
        d2 = d1+f"/{model_key}"
        if os.path.isdir(d2):
            pass
        else:
            os.mkdir(d2)

        fn = d2+f"/{filename}"
        if os.path.isfile(fn) & (recache==False):
            pass
        else:
            #save the results for this site/batch/model
            if batch not in [331]:
                mn = modelname.replace("-epcpn", "")
            else:
                mn = modelname       
            cells, ops = parse_cellid({"batch": batch, "cellid": site})
            xf, ctx = load_model_xform(modelname=mn, batch=batch, cellid=cells[0], eval_model=True)
            extra_epochs = [':'.join([e[0], str(e[1])]) for e in ctx["val"].meta['mask_bins']]
            err = cc_comp(ctx["val"], extra_epochs)

            if perstim:
                pstim_err = {}
                pstim_err["all"] = err
                for s in extra_epochs:
                    err = cc_comp(ctx["val"], s)
                    pstim_err[s] = err
                with open(fn, "wb") as handle:
                    pickle.dump(pstim_err, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(fn, "wb") as handle:
                    pickle.dump(err, handle, protocol=pickle.HIGHEST_PROTOCOL)
