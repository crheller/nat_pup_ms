"""
Summary LV results per site
"""
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/pca_regression_lvs.pickle'
good_sites = True
high_rep_sites = ['TAR010c', 'TAR017b', 
                'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
                'DRX007a.e1:64', 'DRX007a.e65:128', 
                'DRX008b.e1:64', 'DRX008b.e65:128',
                'BOL005c', 'BOL006b']

# load results from pickle file
with open(fn, 'rb') as handle:
    results = pickle.load(handle)

# LOAD VARIANCE EXPLAINED

# get all PC1 var explained
pc1_var = [results[s]['pc_variance'][0] for s in results.keys()]
# get all 1st order pupil var explained
fo_variance = [results[s]['var_1st_order'] for s in results.keys()]
# get all 2nd order pupil var explained
so_variance = [results[s]['var_2nd_order'] for s in results.keys()]

# LOAD COSINE SIMILARITY

# pc1 vs. first order weights
pc1_fow = [abs(results[s]['cos_fow_PC1']) for s in results.keys()]
# pc1 vs. second order weights
pc1_sow = [abs(results[s]['cos_sow_PC1']) for s in results.keys()]
# first order weights vs. second order weights
fow_sow = [abs(results[s]['cos_fow_sow']) for s in results.keys()]

n_neurons = [results[s]['fow'].shape[0] for s in results.keys()]

# pack all results into df for easy plotting
data = np.stack([pc1_var, fo_variance, so_variance, pc1_fow, pc1_sow, fow_sow, n_neurons]).T
df = pd.DataFrame(columns=['pc1_var', 'fo_var', 'so_var', 'pc1_fow', 'pc1_sow', 'fow_sow', 'n_neurons'],
                  index=list(results.keys()), data=data)

if good_sites:
    df = df.loc[high_rep_sites]

# PLOT RESULTS
f, ax = plt.subplots(1, 2, figsize=(8, 4))

sns.stripplot(x='variable', y='value', data=df[['pc1_var', 'fo_var', 'so_var']].melt(), 
                s=5, color='k', ax=ax[0])
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].set_ylabel('Fraction explained variance')
ax[0].set_xticks([0, 1, 2])
ax[0].set_xticklabels(['PC1', '1st-order pupil', '2nd-order pupil'], rotation=45)

sns.stripplot(x='variable', y='value', data=df[['pc1_fow', 'pc1_sow', 'fow_sow']].melt(), 
                s=5, color='k', ax=ax[1])
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].set_ylabel("Cosine Similarity")
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(['PC1 vs. 1st order', 'PC1 vs. 2nd order', '1st vs. 2nd order'], rotation=45)


f.tight_layout()

# connect dots by site
f, ax = plt.subplots(1, 2, figsize=(8, 4))

df[['pc1_var', 'fo_var', 'so_var']].T.plot(ax=ax[0], legend=False, color='lightgrey')
df[['pc1_var', 'fo_var', 'so_var']].mean().plot(ax=ax[0], color='k', lw=2, marker='o')
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].set_ylabel('Fraction explained variance')
ax[0].set_xticks([0, 1, 2])
ax[0].set_xticklabels(['PC1', '1st-order pupil', '2nd-order pupil'], rotation=45)

df[['pc1_fow', 'pc1_sow', 'fow_sow']].T.plot(ax=ax[1], legend=False, alpha=0.7, color='lightgrey')
df[['pc1_fow', 'pc1_sow', 'fow_sow']].mean().plot(ax=ax[1], color='k', lw=2, marker='o')
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].set_ylabel("Cosine Similarity")
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(['PC1 vs. 1st order', 'PC1 vs. 2nd order', '1st vs. 2nd order'], rotation=45)

f.tight_layout()

plt.show()


