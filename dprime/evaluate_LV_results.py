""" 
For various LV model configs (number of LVs / strength of hyperparameter constraint)
evaluate the decoding effects and evaluate noise correlations.
"""

import nems.db as nd
import nems.xforms as xforms
import numpy as np
import charlieTools.plotting as cplt
import json
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import dprime2.load_dprime as ld

sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b', 'BOL005c', 'BOL006b']

raw_model = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7'
lv_models = ['ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV',
                 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV',
                 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.3xR.f3.pred-stategain.4xR.lv_init.i0.xx1.t7-init.f0.t7.pLV']

path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/figures/'
results_path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/results/'
dp_all = ld.load_dprime('dprime_all', xforms_model=raw_model, path=results_path)
dp_sp = ld.load_dprime('dprime_sp_sia', xforms_model=raw_model, path=results_path)

alpha = np.round(np.arange(0, 0.4, 0.1), 2)
results = pd.DataFrame(index=sites, columns=[i[0]+'_'+str(i[1]) for i in itertools.product(['lv1', 'lv2', 'lv3'], alpha)])

for a in alpha:
    if a != alpha[0]:
        modelname1 = lv_models[0].replace('pLV', 'pLV{}'.format(str(a).replace('.', ',')))
        modelname2 = lv_models[1].replace('pLV', 'pLV{}'.format(str(a).replace('.', ',')))
        modelname3 = lv_models[2].replace('pLV', 'pLV{}'.format(str(a).replace('.', ',')))
    else:
        modelname1 = lv_models[0].replace('pLV', 'pLV0')
        modelname2 = lv_models[1].replace('pLV', 'pLV0')
        modelname3 = lv_models[2].replace('pLV', 'pLV0')


    # load lvr dprime results
    sim2_bp_lvr1 = ld.load_dprime('dprime_bp_pr_lvr_sim2_sia_rm1_uMatch', xforms_model=modelname1, path=results_path)
    sim2_sp_lvr1 = ld.load_dprime('dprime_sp_pr_lvr_sim2_sia_rm1_uMatch', xforms_model=modelname1, path=results_path)
    sim2_bp_lvr2 = ld.load_dprime('dprime_bp_pr_lvr_sim2_sia_rm1_uMatch', xforms_model=modelname2, path=results_path)
    sim2_sp_lvr2 = ld.load_dprime('dprime_sp_pr_lvr_sim2_sia_rm1_uMatch', xforms_model=modelname2, path=results_path)
    sim2_bp_lvr3 = ld.load_dprime('dprime_bp_pr_lvr_sim2_sia_rm1_uMatch', xforms_model=modelname3, path=results_path)
    sim2_sp_lvr3 = ld.load_dprime('dprime_sp_pr_lvr_sim2_sia_rm1_uMatch', xforms_model=modelname3, path=results_path)
    df_list = [sim2_bp_lvr1, sim2_sp_lvr1, sim2_bp_lvr2, sim2_sp_lvr2, sim2_bp_lvr3, sim2_sp_lvr3, dp_all, dp_sp]

    for site in sites:
        bp_lv1, sp_lv1, bp_lv2, sp_lv2, bp_lv3, sp_lv3, all_df, sp = ld.filter_df(df_list, dp_all, dp_sp, site=site)
        m1 = ld.get_quad_mask(all_df, sp, 1)
        m2 = ld.get_quad_mask(all_df, sp, 2)
        results.loc[site]['lv1_{}'.format(a)] = (bp_lv1['dprime'][m1 | m2] - sp_lv1['dprime'][m1 | m2]).mean()
        results.loc[site]['lv2_{}'.format(a)] = (bp_lv2['dprime'][m1 | m2] - sp_lv2['dprime'][m1 | m2]).mean()   
        results.loc[site]['lv3_{}'.format(a)] = (bp_lv3['dprime'][m1 | m2] - sp_lv3['dprime'][m1 | m2]).mean()


for site in sites:
    f, ax = plt.subplots(1, 1)

    ax.plot(alpha, results.loc[site][['lv1_0.0', 'lv1_0.1', 'lv1_0.2', 'lv1_0.3']], 'o-', color='grey', label='LV1')
    ax.plot(alpha, results.loc[site][['lv2_0.0', 'lv2_0.1', 'lv2_0.2', 'lv2_0.3']], 'o-', color='red', label='LV2')
    ax.plot(alpha, results.loc[site][['lv3_0.0', 'lv3_0.1', 'lv3_0.2', 'lv3_0.3']], 'o-', color='blue', label='LV3')

    ax.axhline(0, linestyle='--', color='k')
    ax.set_ylabel("Big dprime minus small")
    ax.set_xlabel("alpha")
    ax.set_title(site)

    ax.legend()

    f.savefig(path+site+'.png')

plt.show()