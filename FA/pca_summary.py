# summary of dimensionality, as measured by PCA, relative to nCells, nStimuli, and pupil state

import sys
sys.path.append("/auto/users/hellerc/code/projects/nat_pupil_ms/")
from loader import load_pop_metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 12

from global_settings import CPN_SITES, HIGHR_SITES
import colors

modelname = "factor_analysis_pca"

sites = CPN_SITES + HIGHR_SITES
batches = [331]*len(CPN_SITES) + [322 if s.startswith("BOL")==False else 294 for s in HIGHR_SITES]
columns = ["bp_dim", "sp_dim", "bp_sem", "sp_sem", "nCells", "nStim", "site", "batch"]
metrics = pd.DataFrame(columns=columns, index=range(len(sites)))
for i, (s, b) in enumerate(zip(sites, batches)):
    r = load_pop_metrics(site=s, batch=b, modelname=modelname)
    metrics.iloc[i] = [r["pca_results"]["bp_ta_dim"],
                        r["pca_results"]["sp_ta_dim"],
                        r["sp_dim95"],
                        r["pca_results"]["sp_full_dim_sem"],
                        r["nCells"], 
                        r["nStim"],
                        s, b]


# dimensionality vs. nCells
f, ax = plt.subplots(1, 2, figsize=(10, 5))

m = metrics.batch==331
ax[0].scatter(metrics[m]["nCells"], metrics[m]["bp_dim"], color=colors.LARGE, edgecolor="limegreen", lw=2, s=75)
m = metrics.batch==322
ax[0].scatter(metrics[m]["nCells"], metrics[m]["bp_dim"], color=colors.LARGE, s=50)
ax[0].set_ylabel("bp dim")
ax[0].plot([0, 70], [0, 70], "k--")
m = metrics.batch==331
ax[1].scatter(metrics[m]["nCells"], metrics[m]["sp_dim"], color=colors.SMALL, edgecolor="limegreen", lw=2, s=75)
m = metrics.batch==322
ax[1].scatter(metrics[m]["nCells"], metrics[m]["sp_dim"], color=colors.SMALL, s=50)
ax[1].set_ylabel("sp dim")
#ax[1].plot([0, 70], [0, 70], "k--")
for a in ax:
    a.set_xlabel("n cells")


# small pupil vs. large pupil
f, ax = plt.subplots(1, 1, figsize=(5, 5))

m = metrics.batch==331
ax.scatter((metrics[m]["sp_dim"]+metrics[m]["bp_dim"])/2, metrics[m]["bp_sem"], c=metrics[m]["nStim"], edgecolor="limegreen", lw=2, s=75)
m = metrics.batch==322
ax.scatter((metrics[m]["sp_dim"]+metrics[m]["bp_dim"])/2, metrics[m]["bp_sem"], c=metrics[m]["nStim"], s=50)
ax.set_ylabel("sp dim")
ax.set_xlabel("bp dim")
ax.plot([0, 15], [0, 15], "k--")