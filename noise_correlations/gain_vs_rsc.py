"""
Testing idea that gain effects / noise correlation effects are independent.
Try to predict second order effect using mag. of gain effect.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import os

import load_results as ld

path = '/auto/users/hellerc/results/nat_pupil_ms/first_order_model_results/'

df = pd.concat([pd.read_csv(os.path.join(path,'d_289_pup_sdexp.csv'), index_col=0),
                pd.read_csv(os.path.join(path,'d_294_pup_sdexp.csv'), index_col=0)])
try:
    df['r'] = [np.float(r.strip('[]')) for r in df['r'].values]
    df['r_se'] = [np.float(r.strip('[]')) for r in df['r_se'].values]
except:
    pass

df = df[df.state_chan=='pupil'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se'])
gain = df.loc[:, pd.IndexSlice['gain_mod', 'st.pup']]

rsc_path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations/'
rsc_df = ld.load_noise_correlation('rsc_bal', path=rsc_path)

# add column for the gain of each neuron
g1 = [gain.loc[p.split('_')[0]] for p in rsc_df.index]
g2 = [gain.loc[p.split('_')[1]] for p in rsc_df.index]
rsc_df['g1'] = g1
rsc_df['g2'] = g2
rsc_df['diff'] = rsc_df['sp'] - rsc_df['bp']
mask = (rsc_df['g1'] < .5) & (rsc_df['g2'] < .5)

# heatmap of delta n.c vs. gain of each neuron
f, ax = plt.subplots(1, 3, figsize=(15, 5))

nbins = 15
vm = rsc_df[mask]['all'].std() 
rsc_df[mask].plot.hexbin(x='g1', y='g2', C='all', cmap='PRGn', vmin=-vm, vmax=vm, gridsize=nbins, ax=ax[0])
ax[0].set_xlabel(r"$g_{i}$", fontsize=14)
ax[0].set_ylabel(r"$g_{j}$", fontsize=14)
ax[0].set_title(r"$r_{sc}$")

vm = rsc_df[mask]['diff'].std()
rsc_df[mask].plot.hexbin(x='g1', y='g2', C='diff', cmap='PRGn', vmin=-vm, vmax=vm, gridsize=nbins, ax=ax[1])
ax[1].set_xlabel(r"$g_{i}$", fontsize=14)
ax[1].set_ylabel(r"$g_{j}$", fontsize=14)
ax[1].set_title(r"$\Delta r_{sc}$")

rsc_df[mask].plot.hexbin(x='g1', y='g2', C=None, cmap='Reds', vmin=0, gridsize=nbins, ax=ax[2])
ax[2].set_xlabel(r"$g_{i}$", fontsize=14)
ax[2].set_ylabel(r"$g_{j}$", fontsize=14)
ax[2].set_title(r"Bin Count")

f.tight_layout()

# predictive model. Can we predict delta n.c. from gain?
# predictor are gain_1, gain_2 and interaction term (gain_1 * gain_2)
rsc_df['g1*g2'] = rsc_df['g1'] * rsc_df['g2']
X = sm.add_constant(rsc_df[['g1', 'g2', 'g1*g2']])
y = rsc_df['diff']
model_dnc = sm.OLS(y, X).fit()

X = sm.add_constant(rsc_df[['g1', 'g2', 'g1*g2']])
y = rsc_df['all']
model_all = sm.OLS(y, X).fit()

# plot model results
beta = [r'$0$', r'$g_{i}$', r'$g_{j}$', r'$g_{i}g_{j}$']

f, ax = plt.subplots(1, 1, figsize=(6, 4))

ci = abs(model_dnc.conf_int()[0] - model_dnc.conf_int()[1])
ax.errorbar([-0.1, 0.9, 1.9, 2.9], model_dnc.params.values, yerr=ci.values, 
                        color='r', marker='o', linestyle='none', lw=2, 
                        label=r'$\Delta r_{sc}, R^{2} = %s$' % round(model_dnc.rsquared, 3))

ci = abs(model_all.conf_int()[0] - model_all.conf_int()[1])
ax.errorbar([0.1, 1.1, 2.1, 3.1], model_all.params.values, yerr=ci.values, 
                        color='k', marker='o', linestyle='none', lw=2, 
                        label=r'$r_{sc}, R^{2} = %s$' % round(model_all.rsquared, 3))
ax.axhline(0, linestyle='--', color='grey', lw=2)

ax.legend(frameon=False)

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(beta, fontsize=14)
ax.set_ylabel(r"$\beta$", fontsize=14)
ax.set_xlabel('Predictor')

f.tight_layout()

plt.show()