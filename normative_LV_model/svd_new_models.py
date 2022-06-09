"""
New LV models that allow for different patterns of pupil changes for weak vs.
strong psths. -- fitting 4 matrices (big/small, weak responses, big/small strong resp)

Evaluate decoding predictions / cc predictions for these models
"""
import sys
sys.path.append("/auto/users/hellerc/code/projects/nat_pupil_ms/")
from global_settings import CPN_SITES
import nems_lbhb.projects.nat_pup_decoding.decoding as decoding
import nems.db as nd
from normative_LV_model.helpers import load_delta_dprime

import scipy.stats as ss
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 16

states = [
        'st.pca0.pup+r2+s0,1,2', 
        'st.pca.pup+r2+s0,1,2',
        'st.pca.pup+r2+s1,2', 
        'st.pca.pup+r2+s2', 
        'st.pca.pup+r2', 
        'st.pca0.pup+r2'
]
modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                 "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                 "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
modelnames=[modelname_base.format(s) for s in states]
rawDecoding = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-st.pca.pup+r1-plgsm.p2-aev-rd.resp_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"
modelnames = [rawDecoding] + modelnames
model_keys = [
    "raw",
    "null",
    "pca_only",
    "first_order",
    "indep_noise",
    "LV",
    "LV_shuff_pca"
]
batch = 331
all_sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])

# get results and concat
norm = []
raw = []
for site in all_sites:
    rdf, ndf = load_delta_dprime(site, batch, modelnames=modelnames, columns=model_keys)
    norm.append(ndf)
    raw.append(rdf)

raw_df = pd.concat(raw)
norm_df = pd.concat(norm)

#raw_df = raw_df[raw_df.site.isin(CPN_SITES).values]
#norm_df = norm_df[norm_df.site.isin(CPN_SITES).values]

# add mse
for k in [k for k in model_keys if k!="raw"]:
    raw_df[k+"_sqe"] = np.abs(raw_df[k] - raw_df["raw"])#**2
    norm_df[k+"_sqe"] = np.abs(norm_df[k] - norm_df["raw"])#**2

rg_u = raw_df.groupby("site").mean()
rg_e = raw_df.groupby("site").sem()
ng_u = norm_df.groupby("site").mean()
ng_e = norm_df.groupby("site").sem()

# remove sites where the err for the null model is < err for first order model
bad_sites = rg_u[rg_u["null_sqe"] < np.mean(rg_u[["first_order_sqe", "indep_noise_sqe", "LV_sqe", "LV_shuff_pca_sqe"]], axis=1)].index

raw_df = raw_df[raw_df.site.isin(bad_sites).values==False]
norm_df = norm_df[norm_df.site.isin(bad_sites).values==False]

# recompute grouped dfs after dropping sites
rg_u = raw_df.groupby("site").mean()
rg_e = raw_df.groupby("site").sem()
ng_u = norm_df.groupby("site").mean()
ng_e = norm_df.groupby("site").sem()

# Summary - mean error / cc per site per model

# MSE
f = plt.figure(figsize=(16, 8))

stplt1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
stplt2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
sc1 = plt.subplot2grid((2, 4), (1, 0))
sc2 = plt.subplot2grid((2, 4), (1, 1))
sc3 = plt.subplot2grid((2, 4), (1, 2))
sc4 = plt.subplot2grid((2, 4), (1, 3))

x = np.random.normal(0, 1, ng_u.shape[0])/10
s = 25
for i, k in enumerate(model_keys[1:]):
    stplt1.scatter(x+i, rg_u[k+"_sqe"], label=k, s=s, alpha=0.3)
    stplt2.scatter(x+i, ng_u[k+"_sqe"], label=k, s=s, alpha=0.3)
stplt1.legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left", fontsize=12)
stplt1.errorbar(range(len(model_keys[1:])),
                y=rg_u[[m+"_sqe" for m in model_keys[1:]]].mean().values,
                yerr=rg_u[[m+"_sqe" for m in model_keys[1:]]].sem().values,
                color="k",
                marker="o",
                capsize=2,
                lw=1)
stplt2.errorbar(range(len(model_keys[1:])),
                y=ng_u[[m+"_sqe" for m in model_keys[1:]]].mean().values,
                yerr=ng_u[[m+"_sqe" for m in model_keys[1:]]].sem().values,
                color="k",
                marker="o",
                capsize=2,
                lw=1)
stplt1.set_ylabel(r"MSE (raw $\Delta d'^2$)")
stplt2.set_ylabel(r"MSE (norm. $\Delta d'^2$)")

# first order vs. indep
sc1.scatter(rg_u["first_order_sqe"], rg_u["indep_noise_sqe"], s=15)
mi, ma = (np.min(sc1.get_xlim()+sc1.get_ylim()), np.max(sc1.get_xlim()+sc1.get_ylim()))
sc1.plot([mi, ma], [mi, ma], "k--")
sc1.set_xlabel("First order")
sc1.set_ylabel("Indep noise")
sc1.set_title(f"p={np.round(ss.wilcoxon(rg_u['first_order_sqe'], rg_u['indep_noise_sqe']).pvalue, 3)}")

sc3.scatter(ng_u["first_order_sqe"], ng_u["indep_noise_sqe"], s=15)
mi, ma = np.min(sc3.get_xlim()+sc3.get_ylim()), np.max(sc3.get_xlim()+sc3.get_ylim())
sc3.plot([mi, ma], [mi, ma], "k--")
sc3.set_xlabel("First order")
sc3.set_ylabel("Indep noise")
sc3.set_title(f"p={np.round(ss.wilcoxon(ng_u['first_order_sqe'], ng_u['indep_noise_sqe']).pvalue, 3)}")


# indep vs. LV
sc2.scatter(rg_u["indep_noise_sqe"], rg_u["LV_shuff_pca_sqe"], s=15)
mi, ma = (np.min(sc2.get_xlim()+sc2.get_ylim()), np.max(sc2.get_xlim()+sc2.get_ylim()))
sc2.plot([mi, ma], [mi, ma], "k--")
sc2.set_ylabel("LV")
sc2.set_xlabel("Indep noise")
sc2.set_title(f"p={np.round(ss.wilcoxon(rg_u['LV_shuff_pca_sqe'], rg_u['indep_noise_sqe']).pvalue, 3)}")


sc4.scatter(ng_u["indep_noise_sqe"], ng_u["LV_shuff_pca_sqe"], s=15)
mi, ma = np.min(sc4.get_xlim()+sc4.get_ylim()), np.max(sc4.get_xlim()+sc4.get_ylim())
sc4.plot([mi, ma], [mi, ma], "k--")
sc4.set_ylabel("LV")
sc4.set_xlabel("Indep noise")
sc4.set_title(f"p={np.round(ss.wilcoxon(ng_u['LV_shuff_pca_sqe'], ng_u['indep_noise_sqe']).pvalue, 3)}")


f.tight_layout()


# CC
# need to loop over sites to compute cc for each
rcc_df = pd.DataFrame(columns=model_keys[1:], index=raw_df.site.unique())
ncc_df = pd.DataFrame(columns=model_keys[1:], index=raw_df.site.unique())
for s in raw_df.site.unique():
    rd = raw_df[raw_df.site==s]
    nod = norm_df[norm_df.site==s]
    for k in model_keys[1:]:
        cc = np.corrcoef(rd["raw"], rd[k])[0, 1]
        rcc_df.loc[s, k] = cc
        cc = np.corrcoef(nod["raw"], nod[k])[0, 1]
        ncc_df.loc[s, k] = cc

f = plt.figure(figsize=(16, 8))

stplt1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
stplt2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
sc1 = plt.subplot2grid((2, 4), (1, 0))
sc2 = plt.subplot2grid((2, 4), (1, 1))
sc3 = plt.subplot2grid((2, 4), (1, 2))
sc4 = plt.subplot2grid((2, 4), (1, 3))

s=25
for i, k in enumerate(model_keys[1:]):
    stplt1.scatter(x+i, rcc_df[k], label=k, s=s, alpha=0.3)
    stplt2.scatter(x+i, ncc_df[k], label=k, s=s, alpha=0.3)

stplt1.legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left", fontsize=12)
stplt1.errorbar(range(len(model_keys[1:])),
                y=rcc_df[model_keys[1:]].mean().values,
                yerr=rcc_df[model_keys[1:]].sem().values,
                color="k",
                marker="o",
                capsize=2,
                lw=1)
stplt2.errorbar(range(len(model_keys[1:])),
                y=ncc_df[model_keys[1:]].mean().values,
                yerr=ncc_df[model_keys[1:]].sem().values,
                color="k",
                marker="o",
                capsize=2,
                lw=1)
stplt1.set_ylabel(r"cc (raw $\Delta d'^2$)")
stplt2.set_ylabel(r"cc (norm. $\Delta d'^2$)")

# first order vs. indep
sc1.scatter(rcc_df["first_order"], rcc_df["indep_noise"], s=15)
mi, ma = (np.min(sc1.get_xlim()+sc1.get_ylim()), np.max(sc1.get_xlim()+sc1.get_ylim()))
sc1.plot([mi, ma], [mi, ma], "k--")
sc1.set_xlabel("First order")
sc1.set_ylabel("Indep noise")
sc1.set_title(f"p={np.round(ss.wilcoxon(rcc_df['first_order'], rcc_df['indep_noise']).pvalue, 3)}")

sc3.scatter(ncc_df["first_order"], ncc_df["indep_noise"], s=15)
mi, ma = np.min(sc3.get_xlim()+sc3.get_ylim()), np.max(sc3.get_xlim()+sc3.get_ylim())
sc3.plot([mi, ma], [mi, ma], "k--")
sc3.set_xlabel("First order")
sc3.set_ylabel("Indep noise")
sc3.set_title(f"p={np.round(ss.wilcoxon(ncc_df['first_order'], ncc_df['indep_noise']).pvalue, 3)}")

# indep vs. LV
sc2.scatter(rcc_df["indep_noise"], rcc_df["LV"], s=15)
mi, ma = (np.min(sc2.get_xlim()+sc2.get_ylim()), np.max(sc2.get_xlim()+sc2.get_ylim()))
sc2.plot([mi, ma], [mi, ma], "k--")
sc2.set_ylabel("LV")
sc2.set_xlabel("Indep noise")
sc2.set_title(f"p={np.round(ss.wilcoxon(rcc_df['LV'], rcc_df['indep_noise']).pvalue, 3)}")

sc4.scatter(ncc_df["indep_noise"], ncc_df["LV"], s=15)
mi, ma = np.min(sc4.get_xlim()+sc4.get_ylim()), np.max(sc4.get_xlim()+sc4.get_ylim())
sc4.plot([mi, ma], [mi, ma], "k--")
sc4.set_ylabel("LV")
sc4.set_xlabel("Indep noise")
sc4.set_title(f"p={np.round(ss.wilcoxon(ncc_df['LV'], ncc_df['indep_noise']).pvalue, 3)}")

f.tight_layout()

# bootstrap cc values +/- error across whole population
nboots = 100
cc_norm = np.zeros((100, len(raw_df.site.unique())))
cc_raw = np.zeros((100, len(raw_df.site.unique())))
for bb in range(nboots):
    bidx = np.random.choice(np.arange(0, raw_df.shape[0]), raw_df.shape[0], replace=True)
    rdf = raw_df.iloc[bidx, :]
    ndf = norm_df.iloc[bidx, :]
    for i, k in enumerate(model_keys[1:]):
        cc_norm[bb, i] = np.corrcoef(ndf["raw"], ndf[k])[0, 1]
        cc_raw[bb, i] = np.corrcoef(rdf["raw"], rdf[k])[0, 1]

f, ax = plt.subplots(1, 2, figsize=(10, 5))

for i, k in enumerate(model_keys[1:]):
    u = cc_norm[:, i].mean()
    lq = np.quantile(cc_norm[:, i], 0.025)
    uq = np.quantile(cc_norm[:, i], 0.975)
    ax[0].plot([i, i], [lq, uq], zorder=-1, label=k)
    ax[0].scatter(i, u, s=50, facecolor="white", edgecolor=ax[0].get_lines()[-1].get_color())

    u = cc_raw[:, i].mean()
    lq = np.quantile(cc_raw[:, i], 0.025)
    uq = np.quantile(cc_raw[:, i], 0.975)
    ax[1].plot([i, i], [lq, uq], zorder=-1, label=k)
    ax[1].scatter(i, u, s=50, facecolor="white", edgecolor=ax[1].get_lines()[-1].get_color())

ax[0].set_ylabel(r"cc (norm. ($\Delta d'^2$)", fontsize=12)
ax[1].set_ylabel(r"cc (raw ($\Delta d'^2$)", fontsize=12)
for a in ax:
    a.legend(frameon=False, bbox_to_anchor=(0, 1), loc="lower left", fontsize=12)

f.tight_layout()


# compute hierarchical bootstrap stats
from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob

dd_raw = {s: raw_df[raw_df.site==s]["raw"].values for s in raw_df.site.unique()}
dd_first = {s: raw_df[raw_df.site==s]["first_order"].values for s in raw_df.site.unique()}
dd_ind = {s: raw_df[raw_df.site==s]["indep_noise"].values for s in raw_df.site.unique()}
dd_lv = {s: raw_df[raw_df.site==s]["LV"].values for s in raw_df.site.unique()}
dd_lv0 = {s: raw_df[raw_df.site==s]["LV_shuff_pca"].values for s in raw_df.site.unique()}

# get bootstrapped cc samples
even_samp = True
np.random.seed(123)
bs_first = get_bootstrapped_sample(dd_raw, dd_first, metric="corrcoef", even_sample=even_samp)
np.random.seed(123)
bs_ind = get_bootstrapped_sample(dd_raw, dd_ind, metric="corrcoef", even_sample=even_samp)
np.random.seed(123)
bs_lv = get_bootstrapped_sample(dd_raw, dd_lv, metric="corrcoef", even_sample=even_samp)
np.random.seed(123)
bs_lv0 = get_bootstrapped_sample(dd_raw, dd_lv0, metric="corrcoef", even_sample=even_samp)

bins = np.arange(-0.5, 1, 0.05)
f, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.hist(bs_first, bins=bins, histtype="step", lw=2, label="First order")

ax.hist(bs_ind, bins=bins, histtype="step", lw=2, label="Ind.")

ax.hist(bs_lv, bins=bins, histtype="step", lw=2, label="LV")

ax.hist(bs_lv0, bins=bins, histtype="step", lw=2, label="LV-pca shuff")

ax.legend()

f.tight_layout()


f, ax = plt.subplots(1, 1, figsize=(6, 4))

u = bs_first.mean()
lq = np.quantile(bs_first, 0.025)
uq = np.quantile(bs_first, 0.975)
ax.plot([0, 0], [lq, uq], zorder=-1)
ax.scatter(0, u, edgecolor=ax.get_lines()[-1].get_color(), s=50, facecolor="white", label="first-order")

u = bs_ind.mean()
lq = np.quantile(bs_ind, 0.025)
uq = np.quantile(bs_ind, 0.975)
ax.plot([1, 1], [lq, uq], zorder=-1)
ax.scatter(1, u, edgecolor=ax.get_lines()[-1].get_color(), s=50, facecolor="white", label="indep")

u = bs_lv.mean()
lq = np.quantile(bs_lv, 0.025)
uq = np.quantile(bs_lv, 0.975)
ax.plot([2, 2], [lq, uq], zorder=-1)
ax.scatter(2, u, edgecolor=ax.get_lines()[-1].get_color(), s=50, facecolor="white", label="LV")

u = bs_lv0.mean()
lq = np.quantile(bs_lv0, 0.025)
uq = np.quantile(bs_lv0, 0.975)
ax.plot([3, 3], [lq, uq], zorder=-1)
ax.scatter(3, u, edgecolor=ax.get_lines()[-1].get_color(), s=50, facecolor="white", label="LV pca shuff")

ax.legend(fontsize=12, bbox_to_anchor=(1, 1), loc="upper left")

ax.set_ylabel(r"cc (raw $\Delta d'^2$)")