"""
Need to answer - What predicts the change in decoding as a function of pupil?
    Is is purely dU? Reduction in noise corr? Change in shape of noise corr?
    Answer these q's in order to determine most useful way to present overall 
    decoding results in figure 2.
"""
import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os


loader = decoding.DecodingResults()
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
modelname = 'dprime_pr_jk10_zscore'
n_components = 2
pup_var_pairs = pd.read_csv(path+'high_pvar_stim_combos.csv')

# list of sites with > 10 reps of each stimulus
sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']
site = 'BOL005c'

fn = os.path.join(path, site, modelname+'_PLS.pickle')
results = loader.load_results(fn)

pairs = results.evoked_stimulus_pairs
#pairs = [p for p in pairs if p in pup_var_pairs['combo'].values]

df = results.numeric_results.loc[pd.IndexSlice[pairs, 2], :]

f, ax = plt.subplots(3, 2, figsize=(6, 9))

# plot change in dprime for each stimulus pair
lim = np.max([df['bp_dp'].pow(1/2).max(), df['sp_dp'].pow(1/2).max()])
ax[0, 0].scatter(df['sp_dp'] ** (1/2), df['bp_dp'] ** (1/2), color='k', edgecolor='white', s=25)
ax[0, 0].plot([0, lim], [0, lim], '--', color='grey')
ax[0, 0].axhline(0, linestyle='--', color='grey')
ax[0, 0].axvline(0, linestyle='--', color='grey')
ax[0, 0].set_title(r"$d'$")
ax[0, 0].set_xlabel("Small pupil")
ax[0, 0].set_ylabel("Large pupil")

# plot change in dU magnitude for each stimulus pair
lim = np.max([df['bp_dU_mag'].max(), df['sp_dU_mag'].max()])
ax[0, 1].scatter(df['sp_dU_mag'], df['bp_dU_mag'], color='k', edgecolor='white', s=25)
ax[0, 1].plot([0, lim], [0, lim], '--', color='grey')
ax[0, 1].axhline(0, linestyle='--', color='grey')
ax[0, 1].axvline(0, linestyle='--', color='grey')
ax[0, 1].set_title(r"$|\Delta \mathbf{\mu}|$")
ax[0, 1].set_xlabel('Small pupil')

# plot change in evec 1 snr
bp_evec1 = results.slice_array_results('bp_cos_dU_evec', pairs, 2, idx=[0, 0])[0]
sp_evec1 = results.slice_array_results('sp_cos_dU_evec', pairs, 2, idx=[0, 0])[0]
lim = np.max([bp_evec1.max(), sp_evec1.max()])
ax[1, 0].scatter(sp_evec1, bp_evec1, color='k', edgecolor='white', s=25)
ax[1, 0].plot([0, lim], [0, lim], '--', color='grey')
ax[1, 0].axhline(0, linestyle='--', color='grey')
ax[1, 0].axvline(0, linestyle='--', color='grey')
ax[1, 0].set_title(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{1})|$')
ax[1, 0].set_xlabel("Small pupil")

# plot change in evec 2 snr cos dU
bp_evec2 = results.slice_array_results('bp_cos_dU_evec', pairs, 2, idx=[0, 1])[0]
sp_evec2 = results.slice_array_results('sp_cos_dU_evec', pairs, 2, idx=[0, 1])[0]
lim = np.max([bp_evec2.max(), sp_evec2.max()])
ax[1, 1].scatter(sp_evec2, bp_evec2, color='k', edgecolor='white', s=25)
ax[1, 1].plot([0, lim], [0, lim], '--', color='grey')
ax[1, 1].axhline(0, linestyle='--', color='grey')
ax[1, 1].axvline(0, linestyle='--', color='grey')
ax[1, 1].set_title(r'$|cos(\Delta \mathbf{\mu}, \mathbf{e}_{2})|$')
ax[1, 1].set_xlabel("Small pupil")

# plot change in evec variance ratio
bp_evec1 = results.slice_array_results('bp_evals', pairs, 2, idx=[0])[0]
sp_evec1 = results.slice_array_results('sp_evals', pairs, 2, idx=[0])[0]
bp_evec2 = results.slice_array_results('bp_evals', pairs, 2, idx=[1])[0]
sp_evec2 = results.slice_array_results('sp_evals', pairs, 2, idx=[1])[0]
sp = sp_evec2 / sp_evec1
bp = bp_evec2 / bp_evec1
lim = np.max([bp.max(), sp.max()])
ax[2, 0].scatter(sp, bp, color='k', edgecolor='white', s=25)
ax[2, 0].plot([0, lim], [0, lim], '--', color='grey')
ax[2, 0].axhline(0, linestyle='--', color='grey')
ax[2, 0].axvline(0, linestyle='--', color='grey')
ax[2, 0].set_title(r'$\lambda_{2} / \lambda_{1}$')
ax[2, 0].set_xlabel("Small pupil")

f.tight_layout()

f.canvas.set_window_title(site + ', ' + modelname)

plt.show()