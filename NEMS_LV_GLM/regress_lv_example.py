"""
regress out latent variable for example site, compute noise correlations before and after.
"""

import nems.xforms as xforms
import nems.db as nd
import preprocessing as preproc
from nems_lbhb.preprocessing import create_pupil_mask
import plotting as cplt
import pandas as pd
from itertools import combinations
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

site = 'TAR010c'
# LV constraint
modelname = ['ns.fs4.pup-ld-st.pup-hrc-pbal-psthfr-ev_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.a0:009']
# NC constraint
modelname = ['ns.fs4.pup-ld-st.pup-hrc-apm-psthfr-ev_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC.a0:35']
modelname = ['ns.fs4.pup-ld-st.pup-hrc-apm-psthfr-ev-residual_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf2.p-pupLVbasic.constrNC.a0:05']
modelname2 = ['ns.fs4.pup-ld-st.pup-hrc-psthfr-ev_slogsig.SxR_jk.nf5.p-basic']
batch = 289
cellids = nd.get_batch_cells(batch).cellid
cellid = [[c for c in cellids if site in c][0]]

mp = nd.get_results_file(batch, modelname, cellid).modelpath[0]
mp2 = nd.get_results_file(batch, modelname2, cellid).modelpath[0]

xfspec, ctx = xforms.load_analysis(mp)
xfspec2, ctx2 = xforms.load_analysis(mp2)

# plot summary of fit
ctx2['modelspec'].quickplot()
ctx['modelspec'].quickplot()

rec = ctx['val'].apply_mask(reset_epochs=True).copy()  # raw recording
rec['lv'] = rec['lv']._modified_copy(rec['lv']._data[1, :][np.newaxis, :])
rec1 = ctx2['val'].apply_mask(reset_epochs=True).copy() # first order regression
rec12 = rec.copy()        # first / second order regression

#rec1 = preproc.regress_state(rec1, state_sigs=['pupil'], regress=['pupil'])
#rec12 = preproc.regress_state(rec12, state_sigs=['pupil', 'lv'], regress=['pupil', 'lv'])
psth = preproc.generate_psth(rec12)
data = rec1['resp']._data - rec1['pred']._data + psth['psth']._data
rec1['resp'] = rec1['resp']._modified_copy(data)

psth = preproc.generate_psth(rec12)
data = rec12['resp']._data - rec12['pred']._data + psth['psth']._data
rec12['resp'] = rec12['resp']._modified_copy(data)

# mask pupil, compute noise corr, plot noise corr
f, ax = plt.subplots(1, 2)
for j, (r, s) in enumerate(zip([rec, rec1, rec12], ['raw', '1st', '1st+2nd'])):
    pup_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
    rec_bp = create_pupil_mask(r.copy(), **pup_ops)
    pup_ops['state']='small'
    rec_sp = create_pupil_mask(r.copy(), **pup_ops)
    rec_bp = rec_bp.apply_mask(reset_epochs=True)
    rec_sp = rec_sp.apply_mask(reset_epochs=True)

    eps = np.unique([e for e in rec.epochs.name if 'STIM' in e]).tolist()

    real_dict_all = r['resp'].extract_epochs(eps)
    real_dict_small = rec_sp['resp'].extract_epochs(eps)
    real_dict_big = rec_bp['resp'].extract_epochs(eps)

    real_dict_all = preproc.zscore_per_stim(real_dict_all, d2=real_dict_all)
    real_dict_small = preproc.zscore_per_stim(real_dict_small, d2=real_dict_small)
    real_dict_big = preproc.zscore_per_stim(real_dict_big, d2=real_dict_big)

    nCells = real_dict_all[eps[0]].shape[1]
    for i, k in enumerate(real_dict_all.keys()):
        if i == 0:
            all_data = np.transpose(real_dict_all[k], [1, 0, -1]).reshape(nCells, -1)
            big_data = np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)
            small_data = np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)
        else:
            all_data = np.concatenate((all_data, np.transpose(real_dict_all[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
            small_data = np.concatenate((small_data, np.transpose(real_dict_small[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)
            big_data = np.concatenate((big_data, np.transpose(real_dict_big[k], [1, 0, -1]).reshape(nCells, -1)), axis=-1)

    combos = list(combinations(np.arange(0, nCells), 2))

    df_idx = ["{0}_{1}".format(rec['resp'].chans[i], rec['resp'].chans[j]) for (i, j) in combos]
    cols = ['all', 'p_all', 'bp', 'p_bp', 'sp', 'p_sp', 'site']
    df = pd.DataFrame(columns=cols, index=df_idx)
    for i, pair in enumerate(combos):
        n1 = pair[0]
        n2 = pair[1]
        idx = df_idx[i]

        rr = np.isfinite(all_data[n1, :] + all_data[n2, :])
        cc, pval = ss.pearsonr(all_data[n1, rr], all_data[n2, rr])
        rr = np.isfinite(big_data[n1, :] + big_data[n2, :])
        big_cc, big_pval = ss.pearsonr(big_data[n1, rr], big_data[n2, rr])
        rr = np.isfinite(small_data[n1, :] + small_data[n2, :])
        small_cc, small_pval = ss.pearsonr(small_data[n1, rr], small_data[n2, rr])

        site = idx[:7]
        df.loc[idx, cols] = [cc, pval, big_cc, big_pval, small_cc, small_pval, site]
    
    rnc = df['all'].mean()
    rnc_sem = df['all'].sem()

    dnc = (df['sp'] - df['bp']).mean()
    dnc_sem = (df['sp'] - df['bp']).sem()

    ax[0].bar(j, rnc, yerr=rnc_sem, color='lightgrey', edgecolor='k')
    ax[1].bar(j, dnc, yerr=dnc_sem, color='lightgrey', edgecolor='k')


ax[0].set_ylabel('Overall n.c.')
ax[1].set_ylabel('Delta n.c.')
ax[0].set_xticks([0, 1, 2])
ax[0].set_xticklabels(['raw', '1st', '2nd'])
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(['raw', '1st', '2nd'])
ax[0].set_aspect(cplt.get_square_asp(ax[0]))
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

plt.show()