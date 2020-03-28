"""
Load dprime for all permutation tests. 
Compute mean / sem of dprime in large/small pupil for all 
sites.
"""
import pandas as pd
import numpy as np
import dprime2.load_dprime as ld

permute_modelname = 'dprime_sia_permutation'
perms = np.arange(0, 200)
sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b', 'BOL005c', 'BOL006b']
df = pd.DataFrame(columns=np.sort(sites), index=perms)
dfq1 = pd.DataFrame(columns=np.sort(sites), index=perms)
dfq2 = pd.DataFrame(columns=np.sort(sites), index=perms) 
dfq3 = pd.DataFrame(columns=np.sort(sites), index=perms)
dfq4 = pd.DataFrame(columns=np.sort(sites), index=perms)
dp_all = ld.load_dprime('dprime_all')
dp_sp = ld.load_dprime('dprime_sp_sia')
q1_mask = ld.get_quad_mask(dp_all, dp_sp, 1)
q2_mask = ld.get_quad_mask(dp_all, dp_sp, 2) 
q3_mask = ld.get_quad_mask(dp_all, dp_sp, 3) 
q4_mask = ld.get_quad_mask(dp_all, dp_sp, 4)
for p in perms:
    print('{0} / {1}'.format(p, len(perms)))
    mn_bp = (permute_modelname+str(p)).replace('dprime_', 'dprime_bp_')
    mn_sp = (permute_modelname+str(p)).replace('dprime_', 'dprime_sp_')
    results_bp = ld.load_dprime(mn_bp)
    results_sp = ld.load_dprime(mn_sp)

    diff = results_bp.groupby(by='site').mean()['dprime'] - results_sp.groupby(by='site').mean()['dprime']
    df.loc[p] = diff

    diff1 = results_bp[q1_mask].groupby(by='site').mean()['dprime'] - results_sp[q1_mask].groupby(by='site').mean()['dprime']
    dfq1.loc[p] = diff1

    diff2 = results_bp[q2_mask].groupby(by='site').mean()['dprime'] - results_sp[q2_mask].groupby(by='site').mean()['dprime']
    dfq2.loc[p] = diff2

    diff3 = results_bp[q3_mask].groupby(by='site').mean()['dprime'] - results_sp[q3_mask].groupby(by='site').mean()['dprime']
    dfq3.loc[p] = diff3

    diff4 = results_bp[q4_mask].groupby(by='site').mean()['dprime'] - results_sp[q4_mask].groupby(by='site').mean()['dprime']
    dfq4.loc[p] = diff4

df.to_csv('/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/dprime_control_all.csv')
dfq1.to_csv('/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/dprime_control_q1.csv')
dfq2.to_csv('/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/dprime_control_q2.csv')
dfq3.to_csv('/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/dprime_control_q3.csv')
dfq4.to_csv('/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime2/dprime_control_q4.csv')