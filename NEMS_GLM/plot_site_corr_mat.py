import sys
sys.path.append('/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime')
import dprime_helpers as helpers
import charlieTools.noise_correlations as nc
import charlieTools.plotting as cplt
import nems.xforms as xforms
import nems.db as nd
import json
import matplotlib.pyplot as plt
import numpy as np
import load_results as ld

site = 'AMT019a'
batch = 289
lv_modelstring = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-ev-residual_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC'
lv_modelstring = 'ns.fs4.pup-ld-st.pup-hrc-epsig-apm-pbal-psthfr-ev-residual_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf5.p-pupLVbasic.constrNC'
pupil_modelstring = 'ns.fs4.pup-ld-st.pup-hrc-epsig-apm-pbal-psthfr-ev_slogsig.SxR_jk.nf5.p-basic'
vmin = -.5
vmax = .5
cmap = 'coolwarm'
lv_model = helpers.choose_best_model(site, batch, lv_modelstring, pupil_modelstring, corr_method=2)
#lv_model = 'ns.fs4.pup-ld-st.pup-hrc-epsig-apm-pbal-psthfr-ev-residual_slogsig.SxR-lv.1xR-lvlogsig.2xR_jk.nf2.p-pupLVbasic.constrNC.a0:1'
cellid = [c for c in nd.get_batch_cells(batch).cellid if site in c][0]
d = nd.get_results_file(batch, modelnames=[lv_model], cellids=[cellid])
mp = d['modelpath'][0]

xfspec, ctx = xforms.load_analysis(mp)
nCells = len(ctx['val']['resp'].chans)

f, ax = plt.subplots(3, 2, figsize=(4, 6))

# raw n.c. matrix
rec = ctx['val'].copy()
rec = rec.apply_mask(reset_epochs=True)

rec_bp = rec.copy()
rec_bp['mask'] = rec_bp['p_mask']
rec_sp = rec.copy()
rec_sp['mask'] = rec_sp['p_mask']._modified_copy(~rec_sp['p_mask']._data)

rec_bp = rec_bp.apply_mask(reset_epochs=True)
rec_sp = rec_sp.apply_mask(reset_epochs=True)

eps = np.unique([s for s in rec.epochs.name if 'STIM' in s]).tolist()

real_dict_all = rec['resp'].extract_epochs(eps)
real_dict_small = rec_sp['resp'].extract_epochs(eps)
real_dict_big = rec_bp['resp'].extract_epochs(eps)

df_small = nc.compute_rsc(real_dict_small)
df_big = nc.compute_rsc(real_dict_big)
cc_small = np.zeros((nCells, nCells))
cc_big = np.zeros((nCells, nCells))

for i in df_big.index:
    x, y = i.split('_')
    x, y = (int(x), int(y))

    cc_big[x, y] = df_big.loc[i]['rsc']
    cc_small[x, y] = df_small.loc[i]['rsc']

ax[0, 0].imshow(cc_big, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
ax[0, 0].set_title("Big pupil, raw")

ax[1, 0].imshow(cc_small, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
ax[1, 0].set_title("Small pupil, raw")

ax[2, 0].imshow(cc_small - cc_big, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
ax[2, 0].set_title("Small minus large, mean: {}".format(round((df_small['rsc'] - df_big['rsc']).mean(), 3)))

# corrected n.c. matrix
rec = ctx['val'].copy()
rec = rec.apply_mask(reset_epochs=True)

mod_data = rec['resp']._data - rec['pred']._data + rec['psth_sp']._data
rec['resp'] = rec['resp']._modified_copy(mod_data)

rec_bp = rec.copy()
rec_bp['mask'] = rec_bp['p_mask']
rec_sp = rec.copy()
rec_sp['mask'] = rec_sp['p_mask']._modified_copy(~rec_sp['p_mask']._data)

rec_bp = rec_bp.apply_mask(reset_epochs=True)
rec_sp = rec_sp.apply_mask(reset_epochs=True)

eps = np.unique([s for s in rec.epochs.name if 'STIM' in s]).tolist()

real_dict_all_corr = rec['resp'].extract_epochs(eps)
real_dict_small_corr = rec_sp['resp'].extract_epochs(eps)
real_dict_big_corr = rec_bp['resp'].extract_epochs(eps)

df_small = nc.compute_rsc(real_dict_small_corr)
df_big = nc.compute_rsc(real_dict_big_corr)
cc_small_corr = np.zeros((nCells, nCells))
cc_big_corr = np.zeros((nCells, nCells))

for i in df_big.index:
    x, y = i.split('_')
    x, y = (int(x), int(y))

    cc_big_corr[x, y] = df_big.loc[i]['rsc']
    cc_small_corr[x, y] = df_small.loc[i]['rsc']

ax[0, 1].imshow(cc_big_corr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
ax[0, 1].set_title("Big pupil, corrected")

ax[1, 1].imshow(cc_small_corr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
ax[1, 1].set_title("Small pupil, corrected")

ax[2, 1].imshow(cc_small_corr - cc_big_corr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
ax[2, 1].set_title("Small minus large, mean: {}".format(round((df_small['rsc'] - df_big['rsc']).mean(), 3)))

f.tight_layout()

f, ax = plt.subplots(1, 1)

ax.plot(cc_small - cc_big, cc_small_corr - cc_big_corr, '.', color='grey')
ax.plot([-1, 1], [-1, 1], 'k--')
ax.axhline(0, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')
ax.set_xlabel("Raw n.c. diff")
ax.set_ylabel("Corr. n.c. diff")
ax.set_aspect(cplt.get_square_asp(ax))

# plot the mean change in noise correlation and decoding for each stimulus
#sp_corr = ld.load_dprime_model('dprime_sp_pr_lvr_bal_sia_rm2')
#bp_corr = ld.load_dprime_model('dprime_bp_pr_lvr_bal_sia_rm2')
#sp = ld.load_dprime_model('dprime_bp_bal_sia')
#bp = ld.load_dprime_model('dprime_sp_bal_sia')
delta_nc_raw = []
delta_nc_corr = []
for e in eps:
        bins = real_dict_all_corr[e].shape[-1]
        for b in range(bins):
            # get raw nc diff for this stim
            db = {'ep': real_dict_big[e][:, :, b][:, :, np.newaxis]}
            ds = {'ep': real_dict_small[e][:, :, b][:, :, np.newaxis]}
            bnc = nc.compute_rsc(db)
            snc = nc.compute_rsc(ds)
            delta_nc_raw.append((snc['rsc'] - bnc['rsc']).mean())

            # get corrected nc diff for this stim
            db = {'ep': real_dict_big_corr[e][:, :, b][:, :, np.newaxis]}
            ds = {'ep': real_dict_small_corr[e][:, :, b][:, :, np.newaxis]}
            bnc = nc.compute_rsc(db)
            snc = nc.compute_rsc(ds)
            delta_nc_corr.append((snc['rsc'] - bnc['rsc']).mean())

f, ax = plt.subplots(1, 1)
ax.plot(delta_nc_raw, 'o-', color='k', label='raw')
ax.plot(delta_nc_corr, 'o-', color='grey', label='corrected')
ax.axhline(np.mean(delta_nc_raw), color='k', lw=2)
ax.axhline(np.mean(delta_nc_corr), color='grey', lw=2)
ax.set_ylabel('Delta n.c')
ax.set_xlabel('Stimulus')
ax.axhline(0, linestyle='--', color='r')
ax.legend(fontsize=8)



plt.show()