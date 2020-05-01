import load_results as ld
import matplotlib.pyplot as plt
import numpy as np
import charlieTools.plotting as cplt
import pandas as pd
import scipy.stats as ss

sig_pairs = True
path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations/'
df_all = ld.load_noise_correlation('rsc_bal', path=path)
df_pr = ld.load_noise_correlation('rsc_pr_bal_rm1', path=path)
xf_model = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.2xR.f2.pred-stategain.3xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0,3'
df_lvr = ld.load_noise_correlation('rsc_pr_lvr_bal_rm1', xforms_model=xf_model, path=path)

sig = (df_all['p_bp'] < 0.05) | (df_all['p_sp'] < 0.05)

# plot by site
if sig_pairs:
    all_site = df_all[sig].groupby(by='site').mean()
    pr_site = df_pr[sig].groupby(by='site').mean()
    lvr_site = df_lvr[sig].groupby(by='site').mean()
    df_all = df_all[sig]
    df_pr = df_pr[sig]
    df_lvr = df_lvr[sig]
else:
    all_site = df_all.groupby(by='site').mean()
    pr_site = df_pr.groupby(by='site').mean()
    lvr_site = df_lvr.groupby(by='site').mean()

m = np.max(pd.concat([all_site[['bp', 'sp']], pr_site[['sp', 'bp']], lvr_site[['bp', 'sp']]])).max()
m += 0.02
mi = -0.02
f, ax = plt.subplots(1, 3, figsize=(9, 6))

ax[0].bar(0, all_site['bp'].mean(), yerr= all_site['bp'].sem(), color='orchid', edgecolor='k', label='big')
ax[0].bar(1, all_site['sp'].mean(),  yerr= all_site['sp'].sem(), color='purple', edgecolor='k', label='small')
ax[0].legend(fontsize=8)
ax[0].set_ylim((mi, m))
ax[0].set_title('raw')

for s in all_site.index:
    ax[0].plot([0, 1], [all_site.loc[s]['bp'], all_site.loc[s]['sp']], '-o', color='k')

ax[0].set_aspect(cplt.get_square_asp(ax[0]))


ax[1].bar(0, pr_site['bp'].mean(), yerr= pr_site['bp'].sem(), color='orchid', edgecolor='k', label='big')
ax[1].bar(1, pr_site['sp'].mean(), yerr= pr_site['sp'].sem(), color='purple', edgecolor='k', label='small')
ax[1].legend(fontsize=8)
ax[1].set_ylim((mi, m))
ax[1].set_title('remove pupil')

for s in all_site.index:
    ax[1].plot([0, 1], [pr_site.loc[s]['bp'], pr_site.loc[s]['sp']], '-o', color='k')

ax[1].set_aspect(cplt.get_square_asp(ax[1]))


ax[2].bar(0, lvr_site['bp'].mean(), yerr= lvr_site['bp'].sem(), color='orchid', edgecolor='k', label='big')
ax[2].bar(1, lvr_site['sp'].mean(), yerr= lvr_site['sp'].sem(), color='purple', edgecolor='k', label='small')
ax[2].legend(fontsize=8)
ax[2].set_ylim((mi, m))
ax[2].set_title('remove lv')

for s in all_site.index:
    ax[2].plot([0, 1], [lvr_site.loc[s]['bp'], lvr_site.loc[s]['sp']], '-o', color='k')

ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()


# scatter plot
f, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].plot(all_site['bp'], all_site['sp'], '.', color='k')
ax[0].plot(all_site['bp'].loc['TAR010c'], all_site['sp'].loc['TAR010c'], '.', color='r')
ax[0].plot(all_site['bp'].loc['AMT020a'], all_site['sp'].loc['AMT020a'], '.', color='b')
ax[0].plot([-0.05, 0.25], [-0.05, 0.25], '--', color='grey')
ax[0].set_xlabel('Big', fontsize=8)
ax[0].set_ylabel('Small', fontsize=8)
ax[0].set_title('Raw', fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(pr_site['bp'], pr_site['sp'], '.', color='k')
ax[1].plot(pr_site['bp'].loc['TAR010c'], pr_site['sp'].loc['TAR010c'], '.', color='r')
ax[1].plot(pr_site['bp'].loc['AMT020a'], pr_site['sp'].loc['AMT020a'], '.', color='b')
ax[1].plot([-0.05, 0.25], [-0.05, 0.25], '--', color='grey')
ax[1].set_xlabel('Big', fontsize=8)
ax[1].set_ylabel('Small', fontsize=8)
ax[1].set_title('Pupil regress', fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

ax[2].plot(lvr_site['bp'], lvr_site['sp'], '.', color='k')
ax[2].plot(lvr_site['bp'].loc['TAR010c'], lvr_site['sp'].loc['TAR010c'], '.', color='r')
ax[2].plot(lvr_site['bp'].loc['AMT020a'], lvr_site['sp'].loc['AMT020a'], '.', color='b')
ax[2].plot([-0.05, 0.25], [-0.05, 0.25], '--', color='grey')
ax[2].set_xlabel('Big', fontsize=8)
ax[2].set_ylabel('Small', fontsize=8)
ax[2].set_title('LV + Pupil regress', fontsize=8)
ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()

# plot change in noise correlations per site, before/after correction for each site
f, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].plot(all_site['sp'] - all_site['bp'], pr_site['sp'] - pr_site['bp'], '.', color='k')
ax[0].plot([-0.1, 0.1], [-0.1, 0.1], '--', color='grey')
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].axvline(0, linestyle='--', color='grey')
ax[0].set_xlabel('Raw diff (small minus big)', fontsize=8)
ax[0].set_ylabel('pr diff', fontsize=8)
ax[0].set_title('Raw', fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(all_site['sp'] - all_site['bp'], lvr_site['sp'] - lvr_site['bp'], '.', color='k')
ax[1].plot([-0.1, 0.1], [-0.1, 0.1], '--', color='grey')
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].axvline(0, linestyle='--', color='grey')
ax[1].set_xlabel('Raw diff', fontsize=8)
ax[1].set_ylabel('lv diff', fontsize=8)
ax[1].set_title('Raw', fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

ax[2].plot(pr_site['sp'] - pr_site['bp'], lvr_site['sp'] - lvr_site['bp'], '.', color='k')
ax[2].plot([-0.1, 0.1], [-0.1, 0.1], '--', color='grey')
ax[2].axhline(0, linestyle='--', color='grey')
ax[2].axvline(0, linestyle='--', color='grey')
ax[2].set_xlabel('pr diff', fontsize=8)
ax[2].set_ylabel('lvr diff', fontsize=8)
ax[2].set_title('Raw', fontsize=8)
ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()


# overall
m = 0.08

f, ax = plt.subplots(1, 3, figsize=(9, 6))

ax[0].bar(0, df_all['bp'].mean(), yerr= df_all['bp'].sem(), color='orchid', edgecolor='k', label='big')
ax[0].bar(1, df_all['sp'].mean(),  yerr= df_all['sp'].sem(), color='purple', edgecolor='k', label='small')
ax[0].legend(fontsize=8)
ax[0].set_ylim((0, m))
ax[0].set_title('raw')
ax[0].set_aspect(cplt.get_square_asp(ax[0]))


ax[1].bar(0, df_pr['bp'].mean(), yerr=df_pr['bp'].sem(), color='orchid', edgecolor='k', label='big')
ax[1].bar(1, df_pr['sp'].mean(), yerr=df_pr['sp'].sem(), color='purple', edgecolor='k', label='small')
ax[1].legend(fontsize=8)
ax[1].set_ylim((0, m))
ax[1].set_title('remove pupil')
ax[1].set_aspect(cplt.get_square_asp(ax[1]))


ax[2].bar(0, df_lvr['bp'].mean(), yerr=df_lvr['bp'].sem(), color='orchid', edgecolor='k', label='big')
ax[2].bar(1, df_lvr['sp'].mean(), yerr=df_lvr['sp'].sem(), color='purple', edgecolor='k', label='small')
ax[2].legend(fontsize=8)
ax[2].set_ylim((0, m))
ax[2].set_title('remove lv')
ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()



# compare  delta nc for pr and pr_lvr
f, ax = plt.subplots(1, 2)

ax[0].set_title('Overall')
ax[0].bar(0, (df_pr['sp'] - df_pr['bp']).mean(), 
                yerr=(df_pr['sp'] - df_pr['bp']).sem(), color='lightgrey', edgecolor='k')
ax[0].bar(1, (df_lvr['sp'] - df_lvr['bp']).mean(), 
                yerr=(df_lvr['sp'] - df_lvr['bp']).sem(), color='white', edgecolor='k')
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['pr', 'pr_lvr'])
ax[0].set_ylabel('delta n.c.')

ax[1].set_title('By-site')
ax[1].bar(0, (pr_site['sp'] - pr_site['bp']).mean(), 
            yerr=(pr_site['sp'] - pr_site['bp']).sem(), color='lightgrey', edgecolor='k')
ax[1].bar(1, (lvr_site['sp'] - lvr_site['bp']).mean(), 
            yerr=(lvr_site['sp'] - lvr_site['bp']).sem(), color='white', edgecolor='k')

for s in all_site.index:
    ax[1].plot([0, 1], [pr_site.loc[s]['sp'] - pr_site.loc[s]['bp'], 
                lvr_site.loc[s]['sp'] - lvr_site.loc[s]['bp']], '-o', color='k')



plt.show()
