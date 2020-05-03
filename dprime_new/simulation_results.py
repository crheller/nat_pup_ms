"""
Compare delta dprime for simulated data and real data
"""

import charlieTools.nat_sounds_ms.decoding as decoding

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

loader = decoding.DecodingResults()
path = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/'
modelname = 'dprime_pr_jk10_zscore'
sim1 = 'dprime_sim1_pr_jk10_zscore'
sim2 = 'dprime_sim2_pr_jk10_zscore'
n_components = 2

# list of sites with > 10 reps of each stimulus
sites = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
         'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
         'DRX007a.e1:64', 'DRX007a.e65:128', 
         'DRX008b.e1:64', 'DRX008b.e65:128']
site = 'DRX006b.e65:128'

fn = os.path.join(path, site, modelname+'_TDR.pickle')
results = loader.load_results(fn)
fn = os.path.join(path, site, sim1+'_TDR.pickle')
sim1 = loader.load_results(fn)
fn = os.path.join(path, site, sim2+'_TDR.pickle')
sim2 = loader.load_results(fn)

pairs = results.evoked_stimulus_pairs

f, ax = plt.subplots(1, 1, figsize=(4, 4))

raw_diff = results.get_result('bp_dp', pairs, n_components)[0].pow(1) - results.get_result('sp_dp', pairs, n_components)[0].pow(1)
sim1_diff = sim1.get_result('bp_dp', pairs, n_components)[0].pow(1) - sim1.get_result('sp_dp', pairs, n_components)[0].pow(1)
sim2_diff = sim2.get_result('bp_dp', pairs, n_components)[0].pow(1) - sim2.get_result('sp_dp', pairs, n_components)[0].pow(1)

ax.bar([0, 1, 2], [raw_diff.mean(), sim1_diff.mean(), sim2_diff.mean()],
                yerr=[raw_diff.sem(), sim1_diff.sem(), sim2_diff.sem()],
                edgecolor='k', width=0.5, color=['k', 'gold', 'cyan'], lw=2)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['raw', 'sim1', 'sim2'])
ax.set_ylabel(r"$\Delta d'$")

f.tight_layout()

plt.show()
