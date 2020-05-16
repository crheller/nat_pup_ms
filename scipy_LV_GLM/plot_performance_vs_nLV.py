"""
Plot GLM performance as function of number of LV included in the model
"""

import numpy as np
import matplotlib.pyplot as plt
import GLM.load_results as ld
import plotting as cplt

model_string = 'glm_ev_bal_nLV{}_astep0.02_amax1'
nLVs = np.arange(0, 6)

pred_correlation = dict.fromkeys(nLVs)
nmse = dict.fromkeys(nLVs)

for lv in nLVs:
    results = ld.load_fit(model_string.format(lv))
    sites = results.keys()

    pred_correlation[lv] = np.zeros(len(sites))
    nmse[lv] = np.zeros(len(sites))

    for i, site in enumerate(sites):
        # find the best pred corr for this site (will depend on alpha)
        best = 0
        for k in results[site].keys():
            cc = results[site][k]['cc'].mean()
            if cc > best:
                best = cc
                alpha = k

        pred_correlation[lv][i] = results[site][alpha]['cc'].mean()
        nmse[lv][i] = results[site][alpha]['mse'].mean()


f, ax = plt.subplots(1, 2)
nSites = len(sites)
for lv in nLVs:
    u = pred_correlation[lv].mean()
    sem = pred_correlation[lv].std() / np.sqrt(nSites)
    ax[0].errorbar(lv, u, yerr=sem, color='k', marker='o')
    u = nmse[lv].mean()
    sem = nmse[lv].std() / np.sqrt(nSites)
    ax[1].errorbar(lv, u, yerr=sem, color='r', marker='o')

ax[0].set_xlabel('nLVs')
ax[0].set_ylabel('Prediction correlation')
ax[0].set_aspect(cplt.get_square_asp(ax[0]))
ax[1].set_xlabel('nLVs')
ax[1].set_ylabel('NMSE')
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

plt.show()
