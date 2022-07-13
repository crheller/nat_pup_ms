"""
Summary figures for SVD Paris task.
    * Summary of decoding for FA analysis models.
    * Show that 2nd order important
    * Show that 1st order only less variance
    * Show that diversity is real (variance of delta >> than for null model)

Secondary:
    * FA stats summary (dimensionality etc. as in Yu bridging corr. paper)
"""
import sys

sys.path.append("/auto/users/hellerc/code/projects/nat_pupil_ms")
from path_settings import DPRIME_DIR
from global_settings import CPN_SITES, HIGHR_SITES
import charlieTools.nat_sounds_ms.decoding as decoding
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import os
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 12

def get_max_min(a):
    lo, hi = np.min(a.get_xlim()+a.get_ylim()), np.max(a.get_xlim()+a.get_ylim())
    return lo, hi

loader = decoding.DecodingResults()
modelname = 'dprime_jk10_zscore_globalDDR-sd1-nd1_globalDecodingAxis'
modelname = "dprime_jk10_zscore_fixtdr2-fa"

# list of sites with > 10 reps of each stimulus
sites = CPN_SITES #+ HIGHR_SITES
sites = [s for s in sites if (s!="TNC012a") & (s!="TNC010a")]# & (s!="TNC044a")]
batches = [331]*len(sites) #+ [322]*len(HIGHR_SITES)

# LOAD RAW RESULTS
df = pd.DataFrame()
for batch, site in zip(batches, sites):
    if site in ["BOL005c", "BOL006b"]:
        batch = 294
    print('loaiding site {}'.format(site))
    fn = os.path.join(DPRIME_DIR, str(batch), site, modelname+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results.copy()
    _df = _df.loc[pd.IndexSlice[results.evoked_stimulus_pairs, 2], :]
    _df["site"] = site
    _df["batch"] = batch
    df = pd.concat([df, _df])

df["delta_raw"] = df["bp_dp"] - df["sp_dp"]
df["delta_norm"] = df["delta_raw"] / (df["bp_dp"] + df["sp_dp"])

# LOAD FA RESULTS
df_fa = pd.DataFrame()
fa_keys = ["sim0", "sim1", "sim2", "sim3", "sim4", "sim5", "sim6", "sim7"]
for batch, site in zip(batches, sites):
    print('loaiding site {}'.format(site))
    if site in ["BOL005c", "BOL006b"]:
        batch = 294
    raw = []
    norm = []
    for fa_model in fa_keys:
        model = modelname.replace("_jk10", f"_faModel.{fa_model}_jk10")
        fn = os.path.join(DPRIME_DIR, str(batch), site, model+'_TDR.pickle')
        results = loader.load_results(fn)
        _df = results.numeric_results.copy()
        _df = _df.loc[pd.IndexSlice[results.evoked_stimulus_pairs, 2], :]
        draw = _df["bp_dp"] - _df["sp_dp"]
        dnorm = draw / (_df["bp_dp"] + _df["sp_dp"])
        raw.append(draw.values)
        norm.append(dnorm.values)
    deltas = np.concatenate((np.array(raw), np.array(norm)), axis=0)
    _df = pd.DataFrame(columns=[f"{s}_raw" for s in fa_keys] + [f"{s}_norm" for s in fa_keys],
                            data=deltas.T, index=_df.index)
    _df["site"] = site
    _df["batch"] = batch
    df_fa = pd.concat([df_fa, _df])

merge = pd.concat([df[["dp_opt_test", "mean_pupil_range", "delta_raw", "delta_norm", "site", "batch"]], df_fa], axis=1)
finite = (~merge["delta_raw"].isna() & ~merge["delta_norm"].isna()).values
merge = merge.iloc[np.argwhere(finite).squeeze(), :]
merge = merge.loc[:, ~merge.columns.duplicated()]
mg = merge.groupby(by="site").mean()

# Plot marginal histograms for each relevant FA model

salpha = 0.1
halpha = 0.3
s = 10
sb = 35
xlim = (-1, 1)
ylim = (-1, 1)
hbins = np.arange(xlim[0], xlim[1], 0.05)
xticks = np.arange(xlim[0], xlim[1]+0.25, 0.5)
yticks = np.arange(ylim[0], ylim[1]+0.5, 0.5)

figpath = "/auto/users/hellerc/code/projects/nat_pupil_ms/FA/parisFigs/"

fa_plot = ["sim0", "sim1", "sim3", "sim4", "sim5", "sim6"]
color = ["grey", "tab:orange", "tab:green", "tab:purple", "tab:blue", "tab:red"]
for col, fa_sim in zip(color, fa_plot):
    fig = plt.figure(figsize=(3, 3))

    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax); ax_histx.axis("off")
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax); ax_histy.axis("off")

    cc = np.corrcoef(merge["delta_norm"], merge[f"{fa_sim}_norm"])[0, 1]
    ax.scatter(merge["delta_norm"], merge[f"{fa_sim}_norm"],
                    s=s, alpha=salpha, edgecolor="none",
                    color=col, label=fr"$r={cc:.2}$", rasterized=True)
    ax.scatter(mg["delta_norm"], mg[f"{fa_sim}_norm"], s=sb,
                    edgecolor="k", color=col)
    ax.axhline(0, linestyle="--", color="k"); ax.axvline(0, linestyle="--", color="k");
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.set_xlabel(r"Real $\Delta d'^2$")
    ax.set_ylabel(r"Simulated $\Delta d'^2$")
    ax.legend(frameon=False, bbox_to_anchor=(0, 1), loc="upper left", fontsize=8)


    counts, bins = np.histogram(merge["delta_norm"], bins=hbins)
    sd = merge["delta_norm"].var()
    ax_histx.set_title(fa_sim, fontsize=12)
    ax_histx.hist(bins[:-1], bins, weights=counts/max(counts),
                        color="grey", alpha=halpha,
                        edgecolor="k", lw=1, label=fr"$\sigma^2={sd:.2}$")
    ax_histx.legend(fontsize=8, frameon=False, bbox_to_anchor=(1, 1), loc="upper right")
    ax_histy.hist(bins[:-1], bins, weights=counts/max(counts),
                        edgecolor="grey", lw=1,
                        histtype="step", zorder=-1,
                        orientation="horizontal")
    counts, bins = np.histogram(merge[f"{fa_sim}_norm"], bins=hbins)
    sd = merge[f"{fa_sim}_norm"].var()
    ax_histy.hist(bins[:-1], bins, weights=counts/max(counts),
                        color=col, alpha=halpha,
                        edgecolor=col,
                        lw=1, orientation="horizontal", label=fr"$\sigma^2={sd:.2}$")
    ax_histy.legend(fontsize=8, frameon=False, bbox_to_anchor=(0, 1), loc="upper right")
    figname = figpath + fa_sim + ".svg"
    fig.savefig(figname, dpi=500)


# pairwise statistical tests
cc = np.zeros((len(sites), len(fa_plot[1:])))
d = {k: dict.fromkeys(sites) for k in fa_plot[1:]+["delta_norm"]}
for i, fa_sim in enumerate(fa_plot[1:]):
    for j, site in enumerate(sites):
        mask = merge.site==site
        _cc = np.corrcoef(merge[mask]["delta_norm"], merge[mask][f"{fa_sim}_norm"])[0, 1]
        cc[j, i] = _cc
        d[fa_sim][site] = merge[mask][f"{fa_sim}_norm"]
        if i==0:
            d["delta_norm"][site] = merge[mask]["delta_norm"]

from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob
from matplotlib import colors
np.random.seed(123)
bb1 = get_bootstrapped_sample(d["delta_norm"], d["sim1"], metric="corrcoef", even_sample=True)
bb3 = get_bootstrapped_sample(d["delta_norm"], d["sim3"], metric="corrcoef", even_sample=True)
bb4 = get_bootstrapped_sample(d["delta_norm"], d["sim4"], metric="corrcoef", even_sample=True)
bb5 = get_bootstrapped_sample(d["delta_norm"], d["sim5"], metric="corrcoef", even_sample=True)
bb6 = get_bootstrapped_sample(d["delta_norm"], d["sim6"], metric="corrcoef", even_sample=True)

# convert to r2 (crude)
bb1 = bb1**2
bb3 = bb3**2
bb4 = bb4**2
bb5 = bb5**2
bb6 = bb6**2
# with correlations (Efron correction for bias in distro???)
f, ax = plt.subplots(1, 3, figsize=(9, 3))

for i, (bb, col) in enumerate(zip([bb1, bb3, bb4], ["tab:orange", "tab:green", "tab:purple"])):
    lo = np.quantile(bb**3, 0.025)**(1/3)
    hi = np.quantile(bb**3, 0.975)**(1/3)
    col = list(colors.to_rgba(col))
    col[-1] = 0.5
    ax[0].bar(i, bb.mean(), facecolor=col, edgecolor="k", lw=2)
    ax[0].plot([i, i], [lo, hi], color="k", lw=2)

ax[0].plot([0.1, 0.9], [0.7, 0.7], "k-")
pval = get_direct_prob(bb3, bb1)[0]
ax[0].text(0.1, 0.73, fr"p={pval:.3}", fontsize=8)

ax[0].plot([0.1, 1.9], [0.8, 0.8], "k-")
pval = get_direct_prob(bb4, bb1)[0]
ax[0].text(0.75, 0.83, fr"p={pval:.3}", fontsize=8)

ax[0].set_ylabel(r"$R^2$")
ax[0].set_xticks([])

ax[1].scatter(cc[:, 0]**2, cc[:, 1]**2, c="grey", edgecolor="white", s=50)
lo, hi = get_max_min(ax[1])
ax[1].plot([lo, hi], [lo, hi], "k--")
ax[1].set_xlabel("First-order")
ax[1].set_ylabel("Indep. noise")

ax[2].scatter(cc[:, 0]**2, cc[:, 2]**2, c="grey", edgecolor="white", s=50)
lo, hi = get_max_min(ax[2])
ax[2].plot([lo, hi], [lo, hi], "k--")
ax[2].set_xlabel("First-order")
ax[2].set_ylabel("LV model")

f.tight_layout()

f.savefig(figpath+"sum_Cor_r2.svg", dpi=200)


# without correlations
f, ax = plt.subplots(1, 3, figsize=(9, 3))

for i, (bb, col) in enumerate(zip([bb6, bb5, bb4], ["tab:red", "tab:blue", "tab:purple"])):
    lo = np.quantile(bb**3, 0.025)**(1/3)
    hi = np.quantile(bb**3, 0.975)**(1/3)
    col = list(colors.to_rgba(col))
    col[-1] = 0.5
    ax[0].bar(i, bb.mean(), facecolor=col, edgecolor="k", lw=2)
    ax[0].plot([i, i], [lo, hi], color="k", lw=2)

ax[0].plot([0.1, 0.9], [0.7, 0.7], "k-")
pval = get_direct_prob(bb5, bb6)[0]
ax[0].text(0.1, 0.73, fr"p={pval:.3}", fontsize=8)

ax[0].plot([0.1, 1.9], [0.8, 0.8], "k-")
pval = get_direct_prob(bb4, bb6)[0]
ax[0].text(0.75, 0.83, fr"p={pval:.3}", fontsize=8)

ax[0].set_ylabel(r"$R^2$")
ax[0].set_xticks([])

ax[1].scatter(cc[:, -1]**2, cc[:, -2]**2, c="grey", edgecolor="white", s=50)
lo, hi = get_max_min(ax[1])
ax[1].plot([lo, hi], [lo, hi], "k--")
ax[1].set_xlabel("First-order")
ax[1].set_ylabel("Indep. noise")

ax[2].scatter(cc[:, -1]**2, cc[:, 2]**2, c="grey", edgecolor="white", s=50)
lo, hi = get_max_min(ax[2])
ax[2].plot([lo, hi], [lo, hi], "k--")
ax[2].set_xlabel("First-order")
ax[2].set_ylabel("LV model")

f.tight_layout()

f.savefig(figpath+"sum_noCor_r2.svg", dpi=200)

# raw per site data
sdg = merge.groupby(by="site").std()
cg = merge.groupby(by="site").count()
rg = merge.groupby(by="site").agg({'delta_norm': ['mean', 'min', 'max']})

import statsmodels.api as sm

X = pd.concat([cg[["dp_opt_test"]], mg[["mean_pupil_range"]]], axis=1)
X.columns= ["count", "pupil"]
X["pupil"] = np.sqrt(X["pupil"])
# X = X - X.mean()
# X = X / X.std()
y = sdg["delta_norm"]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

model.summary()

# plot spread at each site and sort by pupil variance
f, ax = plt.subplots(1, 3, figsize=(12, 4))

# raw data
sidx = np.argsort(mg["mean_pupil_range"])
for i, idx in enumerate(sidx):
    site = mg.iloc[idx].name
    data = merge[merge.site==site]["delta_norm"].values
    x = np.random.normal(0, 1, data.shape[0])/10 + i
    ax[0].scatter(x, data, alpha=0.3, s=10, color="grey", edgecolor="none")

ax[0].axhline(0, linestyle="--", color="k")
ax[0].set_ylabel(r"$\Delta d'^2$")
ax[0].set_xlabel("Site (sorted by mean pupil variance)")

# grouped scatter plots
ax[1].scatter(mg["mean_pupil_range"], mg["delta_norm"], s=50, color="grey")
ax[1].set_ylabel(r"Mean $\Delta d'^2$")
ax[1].set_xlabel(r"Pupil variance")
r, p = ss.pearsonr(mg["mean_pupil_range"], mg["delta_norm"])
ax[1].set_title(fr"r={r:.3f}, p={p:.3f}")

ax[2].scatter(mg["mean_pupil_range"], sdg["delta_norm"], s=50, color="grey")
ax[2].set_ylabel(r"Std of $\Delta d'^2$")
ax[2].set_xlabel(r"Pupil variance")
r, p = ss.pearsonr(mg["mean_pupil_range"], sdg["delta_norm"])
ax[2].set_title(fr"r={r:.3f}, p={p:.3f}")

f.tight_layout()

f.savefig(figpath+"pupilVariance_vs_decoding.svg", dpi=200)
