"""
Plot regression coefficients for each panel in figure 4. Then, show the result as insets on each panel.
"""

import colors as color
import ax_labels as alab
from global_settings import ALL_SITES, LOWR_SITES, HIGHR_SITES, NOISE_INTERFERENCE_CUT, DU_MAG_CUT
from path_settings import DPRIME_DIR, PY_FIGURES_DIR, CACHE_PATH

import statsmodels.api as sm
import scipy.stats as ss
import charlieTools.nat_sounds_ms.decoding as decoding
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import seaborn as sns

savefig = False
recache = False # recache dprime results locally
ALL_TRAIN_DATA = False  # use training data for all analysis (even if high rep count site / cross val)
                       # in this case, est = val so doesn't matter if you load _test results or _train results
sites = HIGHR_SITES
path = DPRIME_DIR
fig_fn = PY_FIGURES_DIR+'supp_crossvalidation.svg'
loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_nclvz_fixtdr2'

# ======================================== LOAD THE DATA ===================================
df = []
for site in sites:
    if (site in LOWR_SITES) | (ALL_TRAIN_DATA):
        mn = modelname.replace('_jk10', '_jk1_eev')
    else:
        mn = modelname
    fn = os.path.join(path, site, mn+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=CACHE_PATH, recache=recache)
    _df = results.numeric_results
    stim = results.evoked_stimulus_pairs
    _df = _df.loc[pd.IndexSlice[stim, 2], :]
    _df['cos_dU_evec_test'] = results.slice_array_results('cos_dU_evec_test', stim, 2, idx=[0, 0])[0]
    _df['cos_dU_evec_train'] = results.slice_array_results('cos_dU_evec_train', stim, 2, idx=[0, 0])[0]
    _df['site'] = site
    df.append(_df)

df = pd.concat(df)

# ===================================== GET REGRESSION COEFF. FOR EACH SITE ================================
beta_overall = []
beta_delta = []
ci_overall = []
ci_delta = []
pvals_overall = []
pvals_delta = []
highr_mask = []
rsquared = []
for s in df.site.unique():
    if s in HIGHR_SITES:
        highr_mask.append(True)
    else:
        highr_mask.append(False)

    X = df[df.site==s][['cos_dU_evec_test', 'dU_mag_test']]
    X['dU_mag_test'] = X['dU_mag_test'] - X['dU_mag_test'].mean()
    X['dU_mag_test'] /= X['dU_mag_test'].std()
    X['cos_dU_evec_test'] = X['cos_dU_evec_test'] - X['cos_dU_evec_test'].mean()
    X['cos_dU_evec_test'] /= X['cos_dU_evec_test'].std()

    
    X = sm.add_constant(X)
    X['interaction'] = X['cos_dU_evec_test'] * X['dU_mag_test']
    y = (df[df.site==s]['bp_dp'].values.copy() - df[df.site==s]['sp_dp'].values.copy()) / \
        (df[df.site==s]['bp_dp'].values.copy() + df[df.site==s]['sp_dp'].values.copy())
    y -= y.mean()
    y /= y.std()

    model = sm.OLS(y, X).fit()
    low_ci = model.conf_int().values[:,0]
    high_ci = model.conf_int().values[:,1]
    beta_delta.append(model.params.values)
    ci_delta.append(high_ci - low_ci)
    pvals_delta.append(model.pvalues)
    rsquared.append(model.rsquared)

    y = df[df.site==s]['dp_opt_test']
    y -= y.mean()
    y /= y.std()
    model = sm.OLS(y, X).fit()
    low_ci = model.conf_int().values[:,0]
    high_ci = model.conf_int().values[:,1]
    beta_overall.append(model.params.values)
    ci_overall.append(high_ci - low_ci)
    pvals_overall.append(model.pvalues)


beta_overall = np.stack(beta_overall)
beta_delta = np.stack(beta_delta)
pvals_overall = np.stack(pvals_overall)
pvals_delta = np.stack(pvals_delta)
highr_mask = np.array(highr_mask)

# print statistics for reg. coefficients
print("OVERALL D'")
print("noise intereference beta       mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_overall[:,1]), 
                                                       beta_overall[:,1].std() / np.sqrt(beta_overall.shape[0]), 
                                                       ss.ranksums(beta_overall[:, 1], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_overall[:, 1], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_overall[:, 1]<0.05).sum(), pvals_overall.shape[0]))

      
print("discrimination magnitude beta  mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_overall[:,2]), 
                                                            beta_overall[:,2].std() / np.sqrt(beta_overall.shape[0]), 
                                                       ss.ranksums(beta_overall[:, 2], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_overall[:, 2], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_overall[:, 2]<0.05).sum(), pvals_overall.shape[0]))


      
print("interaction term beta          mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_overall[:, 3]), 
                                                    beta_overall[:, 3].std() / np.sqrt(beta_overall.shape[0]), 
                                                       ss.ranksums(beta_overall[:, 3], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_overall[:, 3], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_overall[:, 3]<0.05).sum(), pvals_overall.shape[0]))

      
      
print("\n")
print("DELTA D'")
print("noise intereference beta       mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_delta[:,1]), 
                                                       beta_delta[:,1].std() / np.sqrt(beta_delta.shape[0]), 
                                                       ss.ranksums(beta_delta[:, 1], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_delta[:, 1], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_delta[:,1]<0.05).sum(), pvals_overall.shape[0]))

print("discrimination magnitude beta  mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_delta[:,2]), 
                                                            beta_delta[:,2].std() / np.sqrt(beta_delta.shape[0]), 
                                                       ss.ranksums(beta_delta[:, 2], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_delta[:, 2], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_delta[:,2]<0.05).sum(), pvals_overall.shape[0]))

      
print("interaction term beta          mean:  {0} \n"
      "                               sem:   {1} \n"
      "                               pval:  {2} \n"
      "                               U stat: {3} \n".format(np.mean(beta_delta[:,3]), 
                                                    beta_delta[:,3].std() / np.sqrt(beta_delta.shape[0]), 
                                                       ss.ranksums(beta_delta[:, 3], np.zeros(beta_delta.shape[0])).pvalue,
                                                       ss.ranksums(beta_delta[:, 3], np.zeros(beta_delta.shape[0])).statistic))
print("{0} / {1} sites significant \n".format((pvals_delta[:,3]<0.05).sum(), pvals_overall.shape[0]))


# ============================== PLOT REGRESSION COEF =============================
fontsize=8
f, ax = plt.subplots(2, 2, figsize=(3, 3), sharey=True)

# Noise interference overall
sns.stripplot(y=beta_overall[:, 1], s=3, edgecolor='white', color='k', ax=ax[0, 0])
ax[0, 0].axhline(0, linestyle='--', color='grey', lw=1)
ax[0, 0].set_ylabel('Slope', fontsize=fontsize)
ax[0, 0].set_title(r"$d'^2$ vs. $|cos(\theta_{|\Delta \mu|, e_1})|$", fontsize=fontsize)

# dU mag overall
sns.stripplot(y=beta_overall[:, 2], s=3, edgecolor='white', color='k', ax=ax[0, 1])
ax[0, 1].axhline(0, linestyle='--', color='grey', lw=1)
ax[0, 1].set_ylabel('Slope', fontsize=fontsize)
ax[0, 1].set_title(r"$d'^2$ vs. $|\Delta \mu|$", fontsize=fontsize)

# Noise interference delta
sns.stripplot(y=beta_delta[:, 1], s=3, edgecolor='white', color='k', ax=ax[1, 0])
ax[1, 0].axhline(0, linestyle='--', color='grey', lw=1)
ax[1, 0].set_ylabel('Slope', fontsize=fontsize)
ax[1, 0].set_title(r"$\Delta d'^2$ vs. $|cos(\theta_{|\Delta \mu|, e_1})|$", fontsize=fontsize)

# dU mag delta
sns.stripplot(y=beta_delta[:, 2], s=3, edgecolor='white', color='k', ax=ax[1, 1])
ax[1, 1].axhline(0, linestyle='--', color='grey', lw=1)
ax[1, 1].set_ylabel('Slope', fontsize=fontsize)
ax[1, 1].set_title(r"$\Delta d'^2$ vs. $|\Delta \mu|$", fontsize=fontsize)

f.tight_layout()

if savefig:
    f.savefig(fig_fn)

# plot as single strip plot
coefs = pd.DataFrame(columns=[r"$d'^2$", r"$d'^2$", r"$\Delta d'^2$", r"$\Delta d'^2$"], 
                    data=np.concatenate((beta_delta[:, [1,2]], beta_overall[:, [1,2]]), axis=1)[:,::-1])
coefs = coefs.melt()
coefs['regressor'] = np.concatenate([['Discrimination Magnitude']*beta_delta.shape[0],
                                ['Noise Interference']*beta_delta.shape[0],
                               ['Discrimination Magnitude']*beta_delta.shape[0],
                               ['Noise Interference']*beta_delta.shape[0]])
f, ax = plt.subplots(1, 1, figsize=(4, 6))
sns.stripplot(y='variable', x='value', data=coefs, hue='regressor', dodge=True, ax=ax,
                                                     palette={'Noise Interference': color.COSTHETA, 'Discrimination Magnitude': color.SIGNAL}, alpha=0.3)
sns.pointplot(y='variable', x='value', data=coefs, hue='regressor', dodge=0.4, join=False, ci=95, ax=ax, errwidth=1, scale=0.7, capsize=0.05,
                                                     palette={'Noise Interference': color.COSTHETA, 'Discrimination Magnitude': color.SIGNAL})
ax.axvline(0, linestyle='--', color='grey')
ax.legend(frameon=False, fontsize=10, title='Regressor')

f.tight_layout()

plt.show()