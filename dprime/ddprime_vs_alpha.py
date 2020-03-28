# plot delta dprime for second order simulations after regressing 
# out latent variable as function of alpha

import dprime2.load_dprime as ld
import numpy as np
import matplotlib.pyplot as plt
import charlieTools.plotting as cplt

# for overall results
xforms_model = 'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:0.sc.rb10'

xforms_models = [
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:0.sc.rb10',
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:01.sc.rb10',
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:02.sc.rb10',
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:03.sc.rb10',
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:04.sc.rb10',
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:05.sc.rb10',
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:06.sc.rb10',
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:07.sc.rb10',
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:08.sc.rb10',
    'ns.fs4.pup-ld-st.pup-hrc-apm-pbal-psthfr-addmeta-aev_puplvmodel.pred.step.pfix.dc.R_pupLVbasic.constrLVonly.af0:09.sc.rb10',
]
results_path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/results/'

sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
        'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
        'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b']

for site in sites:
    print(site)
    # get quadrant masks
    dp_all = ld.load_dprime('dprime_all_bal', xforms_model=xforms_model, path=results_path)
    dp_sp = ld.load_dprime('dprime_sp_bal_sia', xforms_model=xforms_model, path=results_path)
    dp_all = dp_all[dp_all['site']==site]
    dp_sp = dp_sp[dp_sp['site']==site]
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
    q3_mask = (X < Xdiv) & (Y < Ydiv)
    q4_mask = (X > Xdiv) & (Y < Ydiv)


    # load raw (before regression) second order sim results
    bp_raw = ld.load_dprime('dprime_bp_sim2_bal_sia_uMatch', xforms_model=xforms_model, path=results_path)
    sp_raw = ld.load_dprime('dprime_sp_sim2_bal_sia_uMatch', xforms_model=xforms_model, path=results_path)
    bp_raw = bp_raw[bp_raw['site']==site]
    sp_raw = sp_raw[sp_raw['site']==site]
    bp_raw = bp_raw[bp_raw['category']=='sound_sound']
    sp_raw = sp_raw[sp_raw['category']=='sound_sound']
    bp_raw = bp_raw[data_mask]
    sp_raw = sp_raw[data_mask]

    q1_raw_diff = (bp_raw[q1_mask]['dprime'] - sp_raw[q1_mask]['dprime']).mean()
    q2_raw_diff = (bp_raw[q2_mask]['dprime'] - sp_raw[q2_mask]['dprime']).mean()
    q3_raw_diff = (bp_raw[q3_mask]['dprime'] - sp_raw[q3_mask]['dprime']).mean()
    q4_raw_diff = (bp_raw[q4_mask]['dprime'] - sp_raw[q4_mask]['dprime']).mean()

    q1_diffs = np.zeros(len(xforms_models))
    q2_diffs = np.zeros(len(xforms_models))
    q3_diffs = np.zeros(len(xforms_models))
    q4_diffs = np.zeros(len(xforms_models))
    alpha = np.zeros(len(xforms_models))

    for i, xm in enumerate(xforms_models):
        print(str(i) + ' / ' + str(len(xforms_models)))
        bp = ld.load_dprime('dprime_bp_pr_lvr_sim2_bal_sia_rm1_uMatch', xm, results_path)
        sp = ld.load_dprime('dprime_sp_pr_lvr_sim2_bal_sia_rm1_uMatch', xm, results_path)
        
        bp = bp[bp['site']==site]
        sp = sp[sp['site']==site]
        bp = bp[bp['category']=='sound_sound']
        sp = sp[sp['category']=='sound_sound']
        bp = bp[data_mask]
        sp = sp[data_mask]

        q1_diff = (bp[q1_mask]['dprime'] - sp[q1_mask]['dprime']).mean()
        q2_diff = (bp[q2_mask]['dprime'] - sp[q2_mask]['dprime']).mean()
        q3_diff = (bp[q3_mask]['dprime'] - sp[q3_mask]['dprime']).mean()
        q4_diff = (bp[q4_mask]['dprime'] - sp[q4_mask]['dprime']).mean()

        q1_diffs[i] = q1_diff
        q2_diffs[i] = q2_diff
        q3_diffs[i] = q3_diff
        q4_diffs[i] = q4_diff

        a = np.float(xm.split('.af')[-1].split('.sc')[0].replace(':', '.'))
        alpha[i] = a

    f, ax = plt.subplots(2, 2)

    ax[0, 1].plot(alpha, q1_diffs, '-o', color='k', label='lvr diff')
    ax[0, 1].axhline(q1_raw_diff, color='r', linestyle='--', label='raw diff')
    ax[0, 1].axhline(0, color='grey', linestyle='--')
    ax[0, 1].set_xlabel('alpha', fontsize=8)
    ax[0, 1].set_ylabel('2nd order dprime diff', fontsize=8)
    ax[0, 1].legend(fontsize=8, frameon=False)
    ax[0, 1].set_aspect(cplt.get_square_asp(ax[0, 1]))

    ax[0, 0].plot(alpha, q2_diffs, '-o', color='k', label='lvr diff')
    ax[0, 0].axhline(q3_raw_diff, color='r', linestyle='--', label='raw diff')
    ax[0, 0].axhline(0, color='grey', linestyle='--')
    ax[0, 0].set_xlabel('alpha', fontsize=8)
    ax[0, 0].set_ylabel('2nd order dprime diff', fontsize=8)
    ax[0, 0].legend(fontsize=8, frameon=False)
    ax[0, 0].set_aspect(cplt.get_square_asp(ax[0, 0]))

    ax[1, 0].plot(alpha, q3_diffs, '-o', color='k', label='lvr diff')
    ax[1, 0].axhline(q3_raw_diff, color='r', linestyle='--', label='raw diff')
    ax[1, 0].axhline(0, color='grey', linestyle='--')
    ax[1, 0].set_xlabel('alpha', fontsize=8)
    ax[1, 0].set_ylabel('2nd order dprime diff', fontsize=8)
    ax[1, 0].legend(fontsize=8, frameon=False)
    ax[1, 0].set_aspect(cplt.get_square_asp(ax[1, 0]))

    ax[1, 1].plot(alpha, q4_diffs, '-o', color='k', label='lvr diff')
    ax[1, 1].axhline(q4_raw_diff, color='r', linestyle='--', label='raw diff')
    ax[1, 1].axhline(0, color='grey', linestyle='--')
    ax[1, 1].set_xlabel('alpha', fontsize=8)
    ax[1, 1].set_ylabel('2nd order dprime diff', fontsize=8)
    ax[1, 1].legend(fontsize=8, frameon=False)
    ax[1, 1].set_aspect(cplt.get_square_asp(ax[1, 1]))

    f.tight_layout()

    f.canvas.set_window_title(site)

    f.savefig('/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/figures/{}_dprime_alpha.png'.format(site))

plt.show()
