#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:26:49 2018

@author: hellerc
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import nems_lbhb.plots as plots
import charlieTools.xforms_fit as xfit
import charlieTools.plotting as cplt
import copy
import pandas as pd
from sklearn.decomposition import PCA
import nems.xforms as xforms
import nems.db as nd
'''
2nd order raw decoding results per site
        sp_dprime  bp_dprime
site                       
AMT005c  5.657255  6.687819
AMT018a  3.928116  5.728577
AMT019a  2.376777  2.569790
AMT020a  4.403068  5.511751
AMT021b  2.846526  2.935210
AMT023d  3.223473  3.797881
AMT024b  2.616293  3.082475
BRT026c  3.198197  4.243707
BRT034f  4.133248  4.516048
BRT036b  2.074990  2.313242
BRT038b  3.202578  4.812119
BRT039c  3.466530  4.481951
TAR010c  5.062363  5.673480
TAR017b  4.024500  4.736272
bbl086b  3.091243  3.306617
bbl099g  6.330504  8.153558
bbl104h  3.959863  5.145957
'''

cellid = 'TAR010c'
batch = 289
load = False
load_dprime = False

# fast LV with pupil (gain model with sigmoid nonlinearity)
modelname = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev-addmeta_slogsig.SxR-lv.1xR.f.pred-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrLVonly.af0:2.sc'

# single module for LV (testing LV modeling architectures)
#modelname = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev-addmeta_puplvmodel.pred.step.dc.R_jk.nf5.p-pupLVbasic.constrLVonly.af0:3.sc.rb10'
#modelname0 = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev-addmeta_puplvmodel.dc.pupOnly.R_jk.nf5.p-pupLVbasic.constrLVonly.af0:0.sc'

# without jackknifing (or cross validation)
modelname = 'ns.fs4.pup-ld-st.pup-epsig-hrc-apm-pbal-psthfr-ev-addmeta-aev_puplvmodel.pred.step.g.dc.R_pupLVbasic.constrNC.af0:1.sc.rb2'
modelname0 = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev-addmeta-aev_puplvmodel.g.dc.pupOnly.R_pupLVbasic.constrLVonly.af0:0.sc.rb2'
xforms_model = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:0.sc.rb10'
if load:
    c = [c for c in nd.get_batch_cells(batch).cellid if cellid in c][0]
    mp = nd.get_results_file(batch, [modelname], [c]).modelpath[0]
    _, ctx = xforms.load_analysis(mp)
    mp = nd.get_results_file(batch, [modelname0], [c]).modelpath[0]
    _, ctx2 = xforms.load_analysis(mp)
else:
    ctx = xfit.fit_xforms_model(batch, cellid, modelname, save_analysis=False)
    ctx2 = xfit.fit_xforms_model(batch, cellid, modelname0, save_analysis=False)

# plot lv, pupil, PC1 timecourses
if '.g.' in modelname:
    key1 = 'pg'
    key2 = 'lvg'
if '.dc.' in modelname:
    key1 = 'pd'
    key2 = 'lvd'

f, ax = plt.subplots(3, 1)
r = ctx['val'].apply_mask(reset_epochs=True)
p = r['pupil']._data.squeeze()
lv = r['lv']._data.squeeze()
newctx = copy.deepcopy(ctx)
newctx['modelspec'].phi[0][key2] = np.zeros((ctx['modelspec'].phi[0][key2].shape[0], 1))
pred1 = newctx['modelspec'].evaluate(r)

t = np.arange(0, r['resp'].shape[-1])

ax[0].set_title('pupil', fontsize=8)
ax[0].plot(t, p, color='purple')

ax[1].set_title('LV', fontsize=8)
ax[1].scatter(t, lv, cmap='Purples', c=p, vmin=p.min()-0.5)

pca = PCA(n_components=1)
#pca.fit((r['resp']._data - pred1['pred']._data).T)
pca.fit(r['residual']._data.T)
#pc1 = pca.components_ @ (r['resp']._data - pred1['pred']._data)
pc1 = pca.components_ @ (r['residual']._data)
ax[2].scatter(t, pc1.squeeze(), cmap='Purples', c=p, vmin=p.min()-0.5)
ax[2].set_title('PC1', fontsize=8)

f.tight_layout()

# plot pg weights, lvg weights, PC weights
f, ax = plt.subplots(1, 2)

pg = ctx['modelspec'].phi[0][key1].squeeze()
lvg = ctx['modelspec'].phi[0][key2].squeeze()
lve = ctx['modelspec'].phi[0]['lve'].squeeze()
mi = np.min(np.concatenate((lve, lvg)))
ma = np.max(np.concatenate((lve, lvg)))
pc_loading = pca.components_.squeeze()

ax[0].plot(pg, lvg, 'k.')
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].axvline(0, linestyle='--', color='grey')
ax[0].set_xlabel('pupil loading', fontsize=8)
ax[0].set_ylabel('LV loading', fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(pc_loading, lvg, 'k.', label='decoding')
ax[1].plot(pc_loading, lve, 'g.', label='encoding')
ax[1].legend(fontsize=8, frameon=False)
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].axvline(0, linestyle='--', color='grey')
ax[1].plot([mi, ma], [mi, ma], linestyle='--', color='grey')
ax[1].set_xlabel('PC loading', fontsize=8)
ax[1].set_ylabel('LV loading', fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

# compare pupil weights (and dc offset) from first order pupil model 
# to second order LV model
f, ax = plt.subplots(1, 2)

p1g = ctx2['modelspec'].phi[0][key1]
d1 = ctx2['modelspec'].phi[0]['d']
p2g = ctx['modelspec'].phi[0][key1]
d2 = ctx['modelspec'].phi[0]['d']

ax[0].plot(p1g, p2g, '.', color='k')
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].axvline(0, linestyle='--', color='grey')
ax[0].set_xlabel('First order pupil weights', fontsize=8)
ax[0].set_ylabel('Second order pupil weights', fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(d1, d2, '.', color='k')
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].axvline(0, linestyle='--', color='grey')
ax[1].set_xlabel('First order pupil offsets', fontsize=8)
ax[1].set_ylabel('Second order pupil offsets', fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

# compare the pc1 var vs. pupil and lv var vs. pupil

ref_len = r['resp'].extract_epoch('REFERENCE').shape[-1]

lv_var = lv.reshape(-1, ref_len).std(axis=-1)
pc1_var = pc1.reshape(-1, ref_len).std(axis=-1)
p_m = p.reshape(-1, ref_len).mean(axis=-1)

f, ax = plt.subplots(1, 2)

ax[0].plot(p_m, lv_var, 'k.')
cor = np.round(np.corrcoef(p_m, lv_var)[0, 1], 3)
ax[0].set_title('cor coef: {}'.format(cor), fontsize=8)
ax[0].set_ylabel('LV variance', fontsize=8)
ax[0].set_xlabel('Pupil mean', fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(p_m, pc1_var, 'k.')
cor = np.round(np.corrcoef(p_m, pc1_var)[0, 1], 3)
ax[1].set_title('cor coef: {}'.format(cor), fontsize=8)
ax[1].set_ylabel('LV variance', fontsize=8)
ax[1].set_xlabel('Pupil mean', fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

import dprime2.load_dprime as ld
import charlieTools.preprocessing as preproc
import charlieTools.simulate_data as sim
import charlieTools.discrimination_tools as di
import charlieTools.noise_correlations as nc

# get noise corr. for raw data
rec = ctx['val'].apply_mask(reset_epochs=True).copy()
# USE XFORMS PUPIL MASK
rec_bp = rec.copy()
rec_bp['mask'] = rec_bp['p_mask']
rec_sp = rec.copy()
rec_sp['mask'] = rec_sp['p_mask']._modified_copy(~rec_sp['p_mask']._data)

rec_bp = rec_bp.apply_mask(reset_epochs=True)
rec_sp = rec_sp.apply_mask(reset_epochs=True)

# xforms model already has balanced epochs
epochs = np.unique([s for s in rec.epochs.name if 'STIM' in s]).tolist()
real_dict_all = rec['resp'].extract_epochs(epochs)
real_dict_small = rec_sp['resp'].extract_epochs(epochs)
real_dict_big = rec_bp['resp'].extract_epochs(epochs)

raw_bp_nc = nc.compute_rsc(real_dict_big)
raw_sp_nc = nc.compute_rsc(real_dict_small)

# compute model regressed results
rec = ctx['val'].apply_mask(reset_epochs=True).copy()
rec = preproc.regress_state(rec, state_sigs=['pupil', 'lv'], regress=['pupil', 'lv'])
# USE XFORMS PUPIL MASK
rec_bp = rec.copy()
rec_bp['mask'] = rec_bp['p_mask']
rec_sp = rec.copy()
rec_sp['mask'] = rec_sp['p_mask']._modified_copy(~rec_sp['p_mask']._data)

rec_bp = rec_bp.apply_mask(reset_epochs=True)
rec_sp = rec_sp.apply_mask(reset_epochs=True)

# xforms model already has balanced epochs
epochs = np.unique([s for s in rec.epochs.name if 'STIM' in s]).tolist()
corr_dict_all = rec['resp'].extract_epochs(epochs)
corr_dict_small = rec_sp['resp'].extract_epochs(epochs)
corr_dict_big = rec_bp['resp'].extract_epochs(epochs)

reg_bp_nc = nc.compute_rsc(corr_dict_big)
reg_sp_nc = nc.compute_rsc(corr_dict_small)

f, ax = plt.subplots(1, 1)

raw_dnc = raw_sp_nc['rsc'] - raw_bp_nc['rsc']
reg_dnc = reg_sp_nc['rsc'] - reg_bp_nc['rsc']

sig_idx = ['{0}_{1}'.format(x[0], x[1]) for x in np.argwhere(rec.meta['sig_corr_pairs'])]

ax.plot(raw_dnc, reg_dnc, 'k.')
ax.plot(raw_dnc.loc[sig_idx], reg_dnc.loc[sig_idx], 'r.')
ma = np.max(pd.concat([abs(raw_dnc), abs(reg_dnc)]))
mi = -ma
ax.plot([mi, ma], [mi, ma], '--', color='grey')
ax.set_xlabel('raw small minus big nc')
ax.set_ylabel('corr. small minus big nc')

ax.set_aspect(cplt.get_square_asp(ax))

# perform noise correlation analysis per stimulus
dnc_corrected = []
dnc_raw = []
for e in epochs:
    bins = rec['resp'].extract_epoch(e).shape[-1]
    for b in range(bins):
        bp = {'stim': real_dict_big[e][:, :, b][:, :, np.newaxis]}
        sp = {'stim': real_dict_small[e][:, :, b][:, :, np.newaxis]}
        bpc = {'stim': corr_dict_big[e][:, :, b][:, :, np.newaxis]}
        spc = {'stim': corr_dict_small[e][:, :, b][:, :, np.newaxis]}

        bp_nc = nc.compute_rsc(bp)
        sp_nc = nc.compute_rsc(sp)
        bp_nc_corr = nc.compute_rsc(bpc)
        sp_nc_corr = nc.compute_rsc(spc)
        '''
        f, ax = plt.subplots(1, 1)
        ax.plot(sp_nc['rsc'] - bp_nc['rsc'], sp_nc_corr['rsc'] - bp_nc_corr['rsc'], 'k.')
        ax.plot([-1, 1], [-1, 1], '--', color='grey')
        ax.axhline(0, linestyle='--', color='grey')
        ax.axvline(0, linestyle='--', color='grey')
        ax.set_xlabel('raw n.c. diff')
        ax.set_ylabel('corrected n.c. diff')
        ax.set_title("{0}, {1}".format(e, b), fontsize=8)
        ax.set_aspect(cplt.get_square_asp(ax))
        f.canvas.set_window_title("{0}, {1}".format(e, b))
        '''
        dnc_corrected.append((sp_nc_corr['rsc'] - bp_nc_corr['rsc']).mean())
        dnc_raw.append((sp_nc['rsc'] - bp_nc['rsc']).mean())

f, ax = plt.subplots(1, 1)

ax.plot(dnc_raw, 'o-', color='purple', label='raw')
ax.plot(dnc_corrected, 'o-', color='green', label='corrected')
ax.set_ylabel('delta n.c.', fontsize=8)
ax.set_xlabel('stimulus', fontsize=8)
ax.axhline(0, linestyle='--', color='grey')
ax.legend(fontsize=8)



# before / after decoding performance for second order effect
real_dict_big = sim.generate_simulated_trials(real_dict_big, 
                                            r2=real_dict_all, 
                                            keep_stats=[2], N=5000)
real_dict_small = sim.generate_simulated_trials(real_dict_small, 
                                                r2=real_dict_all, 
                                                keep_stats=[2], N=5000)
corr_dict_big = sim.generate_simulated_trials(corr_dict_big, 
                                            r2=corr_dict_all, 
                                            keep_stats=[2], N=5000)
corr_dict_small = sim.generate_simulated_trials(corr_dict_small, 
                                                r2=corr_dict_all, 
                                                keep_stats=[2], N=5000)
bp_raw = di.compute_dprime_from_dicts(real_dict_big, 
                                real_dict_all, 
                                norm=True,
                                LDA=False, 
                                spont_bins=None)
sp_raw = di.compute_dprime_from_dicts(real_dict_small, 
                                real_dict_all, 
                                norm=True,
                                LDA=False, 
                                spont_bins=None)
bp_corrected = di.compute_dprime_from_dicts(corr_dict_big, 
                                corr_dict_all, 
                                norm=True,
                                LDA=False, 
                                spont_bins=None)
sp_corrected = di.compute_dprime_from_dicts(corr_dict_small, 
                                corr_dict_all, 
                                norm=True,
                                LDA=False, 
                                spont_bins=None)

f, ax = plt.subplots(1, 1)

ax.plot(bp_raw['dprime'] - sp_raw['dprime'], bp_corrected['dprime'] - sp_corrected['dprime'], '.')
ax.axhline(0, linestyle='--', color='grey')
ax.axvline(0, linestyle='--', color='grey')
ax.set_xlabel('raw dprime diff')
ax.set_ylabel('corr. dprime diff')
ax.set_aspect(cplt.get_square_asp(ax))
mi, ma = ax.get_xlim()
ax.plot([mi, ma], [mi, ma], '--', color='grey')

plt.show()
'''
# evaluate decoding performance for second order simulations before / after regression

# load raw results
if load_dprime:
    results_path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/results/'
    bp_raw = ld.load_dprime('dprime_bp_sim2_bal_sia_uMatch', xforms_model=xforms_model, path=results_path)
    sp_raw = ld.load_dprime('dprime_sp_sim2_bal_sia_uMatch', xforms_model=xforms_model, path=results_path)
    bp_lvr = ld.load_dprime('dprime_bp_pr_lvr_sim2_bal_sia_rm1_uMatch', xforms_model=modelname, path=results_path)
    sp_lvr = ld.load_dprime('dprime_sp_pr_lvr_sim2_bal_sia_rm1_uMatch', xforms_model=modelname, path=results_path)

    bp_raw = bp_raw[bp_raw['site']==cellid]
    sp_raw = sp_raw[sp_raw['site']==cellid]
    bp_raw = bp_raw[bp_raw['category']=='sound_sound']
    sp_raw = sp_raw[sp_raw['category']=='sound_sound']
    bp_lvr = bp_lvr[bp_lvr['site']==cellid]
    sp_lvr = sp_lvr[sp_lvr['site']==cellid]
    bp_lvr = bp_lvr[bp_lvr['category']=='sound_sound']
    sp_lvr = sp_lvr[sp_lvr['category']=='sound_sound']

else:
    xforms_model = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev-addmeta_puplvmodel.pred.step.dc.R_jk.nf5.p-pupLVbasic.constrLVonly.af0:0.sc'
    results_path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/results/'
    bp_raw = ld.load_dprime('dprime_bp_sim2_bal_sia_uMatch', xforms_model=xforms_model, path=results_path)
    sp_raw = ld.load_dprime('dprime_sp_sim2_bal_sia_uMatch', xforms_model=xforms_model, path=results_path)

    # compute model regressed results
    rec = ctx['val'].apply_mask(reset_epochs=True).copy()
    rec = preproc.regress_state(rec, state_sigs=['pupil', 'lv'], regress=['pupil', 'lv'])
    # USE XFORMS PUPIL MASK
    rec_bp = rec.copy()
    rec_bp['mask'] = rec_bp['p_mask']
    rec_sp = rec.copy()
    rec_sp['mask'] = rec_sp['p_mask']._modified_copy(~rec_sp['p_mask']._data)

    rec_bp = rec_bp.apply_mask(reset_epochs=True)
    rec_sp = rec_sp.apply_mask(reset_epochs=True)

    # xforms model already has balanced epochs
    epochs = np.unique([s for s in rec.epochs.name if 'STIM' in s]).tolist()
    real_dict_all = rec['resp'].extract_epochs(epochs)
    real_dict_small = rec_sp['resp'].extract_epochs(epochs)
    real_dict_big = rec_bp['resp'].extract_epochs(epochs)

    decoding_dict = copy.deepcopy(real_dict_all)

    real_dict_big = sim.generate_simulated_trials(real_dict_big, 
                                                r2=real_dict_all, 
                                                keep_stats=[2], N=5000)
    real_dict_small = sim.generate_simulated_trials(real_dict_small, 
                                                    r2=real_dict_all, 
                                                    keep_stats=[2], N=5000)
    try:
        big_spont_bins = rec_bp['resp'].extract_epoch('PreStimSilence').shape[-1]
        small_spont_bins = rec_sp['resp'].extract_epoch('PreStimSilence').shape[-1]
    except:
        big_spont_bins=small_spont_bins=None

    bp_lvr = di.compute_dprime_from_dicts(real_dict_big, 
                                    decoding_dict, 
                                    norm=True,
                                    LDA=False, 
                                    spont_bins=big_spont_bins)
    sp_lvr = di.compute_dprime_from_dicts(real_dict_small, 
                                    decoding_dict, 
                                    norm=True,
                                    LDA=False, 
                                    spont_bins=small_spont_bins)
    bp_lvr = bp_lvr[bp_lvr['category']=='sound_sound']
    sp_lvr = sp_lvr[sp_lvr['category']=='sound_sound']


# mask top two quadrants:
dp_all = ld.load_dprime('dprime_all_bal', xforms_model=xforms_model, path=results_path)
dp_sp = ld.load_dprime('dprime_sp_bal_sia', xforms_model=xforms_model, path=results_path)
dp_all = dp_all[dp_all['site']==cellid]
dp_sp = dp_sp[dp_sp['site']==cellid]
dp_all = dp_all[dp_all['category']=='sound_sound']
dp_sp = dp_sp[dp_sp['category']=='sound_sound']
xax = 'similarity'
yax = 'pc1_proj_on_dec'
X = dp_all[xax]
Y = abs(dp_sp[yax])
lo_similarity = 0.6
hi_similarity = 0.95
hi_pc1 = 0.55

# define mask to get rid of outliers
data_mask = (X > lo_similarity) & \
            (X < hi_similarity) & \
            (Y < hi_pc1) & \
            (dp_all['dprime'] > 0)  & \
            (dp_all['dprime'] < 15)

Ydiv = np.median([0, hi_pc1])
Xdiv = np.median([lo_similarity, hi_similarity])
X = X[data_mask]
Y = Y[data_mask]
q1_mask = (X > Xdiv) & (Y > Ydiv)
q2_mask = (X < Xdiv) & (Y > Ydiv)

draw = bp_raw['dprime'] - sp_raw['dprime']
dlvr = bp_lvr['dprime'] - sp_lvr['dprime']

#draw = draw[data_mask][(q1_mask | q2_mask)]
#dlvr = dlvr[data_mask][(q1_mask | q2_mask)]
draw = draw[data_mask][q1_mask]
dlvr = dlvr[data_mask][q1_mask]
#Y = Y[q1_mask | q2_mask]
Y = Y[q1_mask]

f, ax = plt.subplots(1, 1)

ax.scatter(draw, dlvr, c=Y, cmap='Purples')
ax.plot(draw.mean(), dlvr.mean(), 'o', markersize=6, color='r')
mi = np.min(pd.concat([draw, dlvr])) 
ma = np.max(pd.concat([draw, dlvr])) 
ax.plot([mi, ma], [mi, ma], '--', color='grey')
ax.axhline(0, linestyle='--', color='grey')
ax.axvline(0, linestyle='--', color='grey')
ax.set_ylabel('2nd order decoding diff after removing LV', fontsize=8)
ax.set_xlabel('2nd order decoding diff of raw data', fontsize=8)
ax.set_aspect(cplt.get_square_asp(ax))
'''
