"""
Plot change in decoding for raw data, pupil regressed data, and lv regressed data
per site, per quadrant
"""
import matplotlib.pyplot as plt
import numpy as np
import dprime2.load_dprime as ld
import charlieTools.plotting as cplt

# global params for loading
site = None
cr = 'rm1'
results_path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/results/'
xforms_lvr_model = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.1xR.f.pred.hp0,5-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7'
xforms_lvr_model = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred.hp0,1-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0'
xforms_pr_model = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7'
xforms_raw_model = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7'

# =================================================================
# load raw overall dprime and small pupil, to define quadrants lims
# data mask (including site mask)
dp_all = ld.load_dprime('dprime_all', xforms_model=xforms_raw_model, path=results_path)
dp_sp = ld.load_dprime('dprime_sp_sia', xforms_model=xforms_raw_model, path=results_path)
# mask data (remove extreme values from df based on hardcoded values defined in function)
dp_all_mask = ld.filter_df(dp_all, dp_all, dp_sp, site=site)
dp_sp_mask = ld.filter_df(dp_sp, dp_all, dp_sp, site=site)

# =================================================================
# Next, load all the data and mask according to the 'data_mask' and 
# site provided above

# load "raw" results
raw_bp = ld.load_dprime('dprime_bp_sia', xforms_model=xforms_raw_model, path=results_path)
sim1_bp = ld.load_dprime('dprime_bp_sim1_sia_uMatch', xforms_model=xforms_raw_model, path=results_path)
sim2_bp = ld.load_dprime('dprime_bp_sim2_sia_uMatch', xforms_model=xforms_raw_model, path=results_path)

raw_sp = ld.load_dprime('dprime_sp_sia', xforms_model=xforms_raw_model, path=results_path)
sim1_sp = ld.load_dprime('dprime_sp_sim1_sia_uMatch', xforms_model=xforms_raw_model, path=results_path)
sim2_sp = ld.load_dprime('dprime_sp_sim2_sia_uMatch', xforms_model=xforms_raw_model, path=results_path)

# load "pupil regressed" results
raw_bp_pr = ld.load_dprime('dprime_bp_pr_sia_{}'.format(cr), xforms_model=xforms_pr_model, path=results_path)
sim1_bp_pr = ld.load_dprime('dprime_bp_pr_sim1_sia_{}_uMatch'.format(cr), xforms_model=xforms_pr_model, path=results_path)
sim2_bp_pr = ld.load_dprime('dprime_bp_pr_sim2_sia_{}_uMatch'.format(cr), xforms_model=xforms_pr_model, path=results_path)

raw_sp_pr = ld.load_dprime('dprime_sp_pr_sia_{}'.format(cr), xforms_model=xforms_pr_model, path=results_path)
sim1_sp_pr = ld.load_dprime('dprime_sp_pr_sim1_sia_{}_uMatch'.format(cr), xforms_model=xforms_pr_model, path=results_path)
sim2_sp_pr = ld.load_dprime('dprime_sp_pr_sim2_sia_{}_uMatch'.format(cr), xforms_model=xforms_pr_model, path=results_path)

# load "lv regressed" results
raw_bp_lvr = ld.load_dprime('dprime_bp_pr_lvr_sia_{}'.format(cr), xforms_model=xforms_lvr_model, path=results_path)
sim1_bp_lvr = ld.load_dprime('dprime_bp_pr_lvr_sim1_sia_{}_uMatch'.format(cr), xforms_model=xforms_lvr_model, path=results_path)
sim2_bp_lvr = ld.load_dprime('dprime_bp_pr_lvr_sim2_sia_{}_uMatch'.format(cr), xforms_model=xforms_lvr_model, path=results_path)

raw_sp_lvr = ld.load_dprime('dprime_sp_pr_lvr_sia_{}'.format(cr), xforms_model=xforms_lvr_model, path=results_path)
sim1_sp_lvr = ld.load_dprime('dprime_sp_pr_lvr_sim1_sia_{}_uMatch'.format(cr), xforms_model=xforms_lvr_model, path=results_path)
sim2_sp_lvr = ld.load_dprime('dprime_sp_pr_lvr_sim2_sia_{}_uMatch'.format(cr), xforms_model=xforms_lvr_model, path=results_path)

df_list = [
    raw_bp, sim1_bp, sim2_bp, raw_sp, sim1_sp, sim2_sp,
    raw_bp_pr, sim1_bp_pr, sim2_bp_pr, raw_sp_pr, sim1_sp_pr, sim2_sp_pr,
    raw_bp_lvr, sim1_bp_lvr, sim2_bp_lvr, raw_sp_lvr, sim1_sp_lvr, sim2_sp_lvr,
    dp_all, dp_sp
]

raw_bp, sim1_bp, sim2_bp, raw_sp, sim1_sp, sim2_sp, \
    raw_bp_pr, sim1_bp_pr, sim2_bp_pr, raw_sp_pr, sim1_sp_pr, sim2_sp_pr, \
    raw_bp_lvr, sim1_bp_lvr, sim2_bp_lvr, raw_sp_lvr, sim1_sp_lvr, sim2_sp_lvr, dp_all, dp_sp = ld.filter_df(df_list, dp_all, dp_sp, site=site)

# =================================================================
# create plots


# plot 1 - plot dprime change as function of state correction for each
# simulation
f, ax = plt.subplots(2, 2)

for a, q in zip(ax.flatten(), [2, 1, 3, 4]):
    m = ld.get_quad_mask(dp_all, dp_sp, q)
    a.plot([0, 1, 2],
                [
                    (raw_bp['dprime'][m] - raw_sp['dprime'][m]).mean(),
                    (raw_bp_pr['dprime'][m] - raw_sp_pr['dprime'][m]).mean(),
                    (raw_bp_lvr['dprime'][m] - raw_sp_lvr['dprime'][m]).mean()
                ],
                color='k',
                marker='o',
                markersize=5,
                label="'raw' data")
    a.errorbar([0, 1, 2],
            [
                (raw_bp['dprime'][m] - raw_sp['dprime'][m]).mean(),
                (raw_bp_pr['dprime'][m] - raw_sp_pr['dprime'][m]).mean(),
                (raw_bp_lvr['dprime'][m] - raw_sp_lvr['dprime'][m]).mean()
            ],
            yerr=[
                (raw_bp['dprime'][m] - raw_sp['dprime'][m]).sem(),
                (raw_bp_pr['dprime'][m] - raw_sp_pr['dprime'][m]).sem(),
                (raw_bp_lvr['dprime'][m] - raw_sp_lvr['dprime'][m]).sem()
            ],
            color='k')

    a.plot([0, 1, 2],
                [
                    (sim1_bp['dprime'][m] - sim1_sp['dprime'][m]).mean(),
                    (sim1_bp_pr['dprime'][m] - sim1_sp_pr['dprime'][m]).mean(),
                    (sim1_bp_lvr['dprime'][m] - sim1_sp_lvr['dprime'][m]).mean()
                ],
                color='gold',
                marker='o',
                markersize=5,
                label="1st-sim")
    a.errorbar([0, 1, 2],
            [
                (sim1_bp['dprime'][m] - sim1_sp['dprime'][m]).mean(),
                (sim1_bp_pr['dprime'][m] - sim1_sp_pr['dprime'][m]).mean(),
                (sim1_bp_lvr['dprime'][m] - sim1_sp_lvr['dprime'][m]).mean()
            ],
            yerr=[
                (sim1_bp['dprime'][m] - sim1_sp['dprime'][m]).sem(),
                (sim1_bp_pr['dprime'][m] - sim1_sp_pr['dprime'][m]).sem(),
                (sim1_bp_lvr['dprime'][m] - sim1_sp_lvr['dprime'][m]).sem()
            ],
            color='gold')

    a.plot([0, 1, 2],
                [
                    (sim2_bp['dprime'][m] - sim2_sp['dprime'][m]).mean(),
                    (sim2_bp_pr['dprime'][m] - sim2_sp_pr['dprime'][m]).mean(),
                    (sim2_bp_lvr['dprime'][m] - sim2_sp_lvr['dprime'][m]).mean()
                ],
                color='cyan',
                marker='o',
                markersize=5,
                label="2nd-sim")
    a.errorbar([0, 1, 2],
            [
                (sim2_bp['dprime'][m] - sim2_sp['dprime'][m]).mean(),
                (sim2_bp_pr['dprime'][m] - sim2_sp_pr['dprime'][m]).mean(),
                (sim2_bp_lvr['dprime'][m] - sim2_sp_lvr['dprime'][m]).mean()
            ],
            yerr=[
                (sim2_bp['dprime'][m] - sim2_sp['dprime'][m]).sem(),
                (sim2_bp_pr['dprime'][m] - sim2_sp_pr['dprime'][m]).sem(),
                (sim2_bp_lvr['dprime'][m] - sim2_sp_lvr['dprime'][m]).sem()
            ],
            color='cyan')
    a.axhline(0, linestyle='--', color='grey')
    a.legend(fontsize=8, frameon=False)
    a.set_xticks([0, 1, 2])
    a.set_xticklabels(['raw', 'rem. pup', 'rem. LV+pup'], rotation=45, fontsize=8)
    a.set_ylabel('Dprime big minus small', fontsize=8)
    a.set_aspect(cplt.get_square_asp(a))

f.canvas.set_window_title(site)
f.tight_layout()

'''
# combine top two quadrants
f, ax = plt.subplots(1, 1)

for a, m in zip([ax], [q2_mask | q1_mask]):
    a.plot([0, 1, 2],
                [
                    (raw_bp['dprime'][m] - raw_sp['dprime'][m]).mean(),
                    (raw_bp_pr['dprime'][m] - raw_sp_pr['dprime'][m]).mean(),
                    (raw_bp_lvr['dprime'][m] - raw_sp_lvr['dprime'][m]).mean()
                ],
                color='k',
                marker='o',
                markersize=5,
                label="'raw' data")
    a.errorbar([0, 1, 2],
            [
                (raw_bp['dprime'][m] - raw_sp['dprime'][m]).mean(),
                (raw_bp_pr['dprime'][m] - raw_sp_pr['dprime'][m]).mean(),
                (raw_bp_lvr['dprime'][m] - raw_sp_lvr['dprime'][m]).mean()
            ],
            yerr=[
                (raw_bp['dprime'][m] - raw_sp['dprime'][m]).sem(),
                (raw_bp_pr['dprime'][m] - raw_sp_pr['dprime'][m]).sem(),
                (raw_bp_lvr['dprime'][m] - raw_sp_lvr['dprime'][m]).sem()
            ],
            color='k')

    a.plot([0, 1, 2],
                [
                    (sim1_bp['dprime'][m] - sim1_sp['dprime'][m]).mean(),
                    (sim1_bp_pr['dprime'][m] - sim1_sp_pr['dprime'][m]).mean(),
                    (sim1_bp_lvr['dprime'][m] - sim1_sp_lvr['dprime'][m]).mean()
                ],
                color='gold',
                marker='o',
                markersize=5,
                label="1st-sim")
    a.errorbar([0, 1, 2],
            [
                (sim1_bp['dprime'][m] - sim1_sp['dprime'][m]).mean(),
                (sim1_bp_pr['dprime'][m] - sim1_sp_pr['dprime'][m]).mean(),
                (sim1_bp_lvr['dprime'][m] - sim1_sp_lvr['dprime'][m]).mean()
            ],
            yerr=[
                (sim1_bp['dprime'][m] - sim1_sp['dprime'][m]).sem(),
                (sim1_bp_pr['dprime'][m] - sim1_sp_pr['dprime'][m]).sem(),
                (sim1_bp_lvr['dprime'][m] - sim1_sp_lvr['dprime'][m]).sem()
            ],
            color='gold')

    a.plot([0, 1, 2],
                [
                    (sim2_bp['dprime'][m] - sim2_sp['dprime'][m]).mean(),
                    (sim2_bp_pr['dprime'][m] - sim2_sp_pr['dprime'][m]).mean(),
                    (sim2_bp_lvr['dprime'][m] - sim2_sp_lvr['dprime'][m]).mean()
                ],
                color='cyan',
                marker='o',
                markersize=5,
                label="2nd-sim")
    a.errorbar([0, 1, 2],
            [
                (sim2_bp['dprime'][m] - sim2_sp['dprime'][m]).mean(),
                (sim2_bp_pr['dprime'][m] - sim2_sp_pr['dprime'][m]).mean(),
                (sim2_bp_lvr['dprime'][m] - sim2_sp_lvr['dprime'][m]).mean()
            ],
            yerr=[
                (sim2_bp['dprime'][m] - sim2_sp['dprime'][m]).sem(),
                (sim2_bp_pr['dprime'][m] - sim2_sp_pr['dprime'][m]).sem(),
                (sim2_bp_lvr['dprime'][m] - sim2_sp_lvr['dprime'][m]).sem()
            ],
            color='cyan')
    a.axhline(0, linestyle='--', color='grey')
    a.legend(fontsize=8, frameon=False)
    a.set_xticks([0, 1, 2])
    a.set_xticklabels(['raw', 'rem. pup', 'rem. LV+pup'], rotation=45, fontsize=8)
    a.set_ylabel('Dprime big minus small', fontsize=8)
    a.set_aspect(cplt.get_square_asp(a))

f.canvas.set_window_title(site)
f.tight_layout()
'''

plt.show()