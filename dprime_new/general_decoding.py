"""
New decoding / ddr strategy as of 10.06.2022
    Define decoding space and / or decoding axis across all stimuli for each estimation set.
    Idea is to see how a "general" purpose decoder is affect by pupil / noise
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
sites = [s for s in sites if (s!="TNC010a") & (s!="TNC044a")]
batches = [331]*len(sites) #+ [322]*len(HIGHR_SITES)

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

# load standard ddr decoder for comparison
df_std = pd.DataFrame()
modelname0 = "dprime_jk10_zscore_fixtdr2-fa"
for batch, site in zip(batches, sites):
    print('loaiding site {}'.format(site))
    if site in ["BOL005c", "BOL006b"]:
        batch = 294
    fn = os.path.join(DPRIME_DIR, str(batch), site, modelname0+'_TDR.pickle')
    results = loader.load_results(fn)
    _df = results.numeric_results.copy()
    _df = _df.loc[pd.IndexSlice[results.evoked_stimulus_pairs, 2], :]
    _df["site"] = site
    _df["batch"] = batch
    df_std = pd.concat([df_std, _df])

df_std["delta_raw"] = df_std["bp_dp"] - df_std["sp_dp"]
df_std["delta_norm"] = df_std["delta_raw"] / (df_std["bp_dp"] + df_std["sp_dp"])


# scatter plot of overall dprime, ddr vs. general decoder
# scatter plot of delta dprime, ddr vs. general decoding
f, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].scatter(df["dp_opt_test"], df_std["dp_opt_test"], 
                    s=10, edgecolor="none", alpha=0.5)
ax[0].set_title("Overall dprime")

ax[1].scatter(df["delta_norm"], df_std["delta_norm"],
                    s=10, edgecolor="none", alpha=0.5)
ax[1].set_title("Norm delta")

ax[2].scatter(df["delta_raw"], df_std["delta_raw"],
                    s=10, edgecolor="none", alpha=0.5)
ax[2].set_title("Raw delta")
for a in ax:
    lo, hi = get_max_min(a)
    a.plot([lo, hi], [lo, hi], "k--")
    a.set_xlabel("'Dumb', general decoder")
    a.set_ylabel("'Standard' dDR")
f.tight_layout()

# Scatter of bp vs. sp and distro of delta
f, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].scatter(df["sp_dp"], df["bp_dp"], s=10, alpha=0.5, edgecolor="none")
lo, hi = np.min(ax[0].get_xlim()+ax[0].get_ylim()), np.max(ax[0].get_xlim()+ax[0].get_ylim())
ax[0].plot([lo, hi], [lo, hi], "k--")
ax[0].set_xlabel("Small pupil")
ax[0].set_ylabel("Large pupil")

ax[1].hist(df["delta_raw"], bins=np.arange(-25, 25, 1),
            edgecolor="k", histtype="stepfilled", alpha=0.5)
ax[1].set_xlabel(r"$\Delta d'^2$ (raw)")

ax[2].hist(df["delta_norm"], bins=np.arange(-1, 1, 0.05),
            edgecolor="k", histtype="stepfilled", alpha=0.5)
ax[2].set_xlabel(r"$\Delta d'^2$ (norm)")


# load corresponding FA simulations for this model
# Answer Q: Do we need noise to predict delta dprimes?
df_fa = pd.DataFrame()
fa_keys = ["sim1", "sim2", "sim3", "sim4"]
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
# merge = merge[merge.dp_opt_test<25]
# merge = merge[merge.dp_opt_test>5]
merge = merge.loc[:, ~merge.columns.duplicated()]

# FA deltas vs. raw delta
f, ax = plt.subplots(2, 4, figsize=(20, 10))

for i, k in enumerate(["norm", "raw"]):
    x, y = merge[f"delta_{k}"].values, merge[f"sim1_{k}"].values
    ax[i, 0].scatter(x, y,
                        s=10, edgecolor="none", alpha=0.3)
    ff = np.isfinite(x) & np.isfinite(y)
    mse = np.std(x[ff]-y[ff])
    cc = np.corrcoef(x[ff], y[ff])[0, 1]
    ax[i, 0].set_title(f"first order, r={cc:.3f}, mse={mse:.3f}")


    x, y = merge[f"delta_{k}"].values, merge[f"sim2_{k}"].values
    ax[i, 1].scatter(x, y,
                        s=10, edgecolor="none", alpha=0.3)
        
    ff = np.isfinite(x) & np.isfinite(y)
    mse = np.std(x[ff]-y[ff])
    cc = np.corrcoef(x[ff], y[ff])[0, 1]
    ax[i, 1].set_title(f"ind. fix cov., r={cc:.3f}, mse={mse:.3f}")

    x, y = merge[f"delta_{k}"].values, merge[f"sim3_{k}"].values
    ax[i, 2].scatter(x, y,
                        s=10, edgecolor="none", alpha=0.3)
    ff = np.isfinite(x) & np.isfinite(y)
    mse = np.std(x[ff]-y[ff])
    cc = np.corrcoef(x[ff], y[ff])[0, 1]
    ax[i, 2].set_title(f"ind. fix cc., r={cc:.3f}, mse={mse:.3f}")

    x, y = merge[f"delta_{k}"].values, merge[f"sim4_{k}"].values
    ax[i, 3].scatter(x, y,
                        s=10, edgecolor="none", alpha=0.3)
    ff = np.isfinite(x) & np.isfinite(y)
    mse = np.std(x[ff]-y[ff])
    cc = np.corrcoef(x[ff], y[ff])[0, 1]
    ax[i, 3].set_title(f"full model, r={cc:.3f}, mse={mse:.3f}")

for a in ax.flatten():
    lo, hi = get_max_min(a)
    a.plot([lo, hi], [lo, hi], "k--")
    a.set_xlabel("raw")
    a.set_ylabel("FA")
f.tight_layout()

# directly compare correlation coefficients / mse per site for first order vs. indep
# and indep vs. indep2, indep vs. LV, indep2 vs. LV
pairs = [
    ["sim1", "sim2"],
    ["sim2", "sim3"],
    ["sim2", "sim4"],
    ["sim3", "sim4"]
]
pairs = [
    ["sim7", "sim6"],
    ["sim6", "sim5"],
    ["sim5", "sim4"],
    ["sim1", "sim4"]
]
f, ax = plt.subplots(2, 4, figsize=(20, 10))

for i, p in enumerate(pairs):
    e1 = []
    cc1 = []
    e2 = []
    cc2 = []
    pvar = []
    for s in df.site.unique():
        mask = (merge.site==s).values
        e1.append(np.std(merge[mask]["delta_norm"].values-merge[mask][f"{p[0]}_norm"].values))
        e2.append(np.std(merge[mask]["delta_norm"].values-merge[mask][f"{p[1]}_norm"].values))
        cc1.append(np.corrcoef(merge[mask]["delta_norm"].values, merge[mask][f"{p[0]}_norm"].values)[0, 1])
        cc2.append(np.corrcoef(merge[mask]["delta_norm"].values, merge[mask][f"{p[1]}_norm"].values)[0, 1])
        pvar.append(merge[mask]["mean_pupil_range"].mean())
    ax[0, i].scatter(e1, e2, s=75, edgecolor="white", c=pvar, cmap="plasma")
    ax[1, i].scatter(cc1, cc2, s=75, edgecolor="white", c=pvar, cmap="plasma")
    ax[0, i].set_xlabel(p[0]); ax[0, i].set_ylabel(p[1])
    ax[1, i].set_xlabel(p[0]); ax[1, i].set_ylabel(p[1])

    mse_pval = ss.wilcoxon(e1, e2).pvalue
    cc_pval = ss.wilcoxon(cc1, cc2).pvalue
    ax[0, i].set_title(f"MSE, p={mse_pval:.3f}")
    ax[1, i].set_title(f"CC, p={cc_pval:.3f}")

for a in ax.flatten():
    lo, hi = get_max_min(a)
    a.plot([lo, hi], [lo, hi], "k--")

f.tight_layout()

f, ax = plt.subplots(2, 4, figsize=(20, 10))
for i, p in enumerate(pairs):
    e1 = []
    cc1 = []
    e2 = []
    cc2 = []
    pvar = []
    for s in df.site.unique():
        mask = (merge.site==s).values
        e1.append(np.abs(merge[mask]["delta_raw"].values-merge[mask][f"{p[0]}_raw"].values).mean())
        e2.append(np.abs(merge[mask]["delta_raw"].values-merge[mask][f"{p[1]}_raw"].values).mean())
        cc1.append(np.corrcoef(merge[mask]["delta_raw"].values, merge[mask][f"{p[0]}_raw"].values)[0, 1])
        cc2.append(np.corrcoef(merge[mask]["delta_raw"].values, merge[mask][f"{p[1]}_raw"].values)[0, 1])
        pvar.append(merge[mask]["mean_pupil_range"].mean())
    ax[0, i].scatter(e1, e2, s=75, edgecolor="white", c=pvar, cmap="plasma")
    ax[1, i].scatter(cc1, cc2, s=75, edgecolor="white", c=pvar, cmap="plasma")
    ax[0, i].set_xlabel(p[0]); ax[0, i].set_ylabel(p[1])
    ax[1, i].set_xlabel(p[0]); ax[1, i].set_ylabel(p[1])

    mse_pval = ss.wilcoxon(e1, e2).pvalue
    cc_pval = ss.wilcoxon(cc1, cc2).pvalue
    ax[0, i].set_title(f"MSE, p={mse_pval:.3f}")
    ax[1, i].set_title(f"CC, p={cc_pval:.3f}")

for a in ax.flatten():
    lo, hi = get_max_min(a)
    a.plot([lo, hi], [lo, hi], "k--")


mg = merge.groupby(by="site").mean()
f, ax = plt.subplots(1, 6, figsize=(18, 3), sharex=True, sharey=True)

ax[0].scatter(mg["delta_norm"], mg["sim0_norm"], c=mg["mean_pupil_range"], cmap="plasma")
ax[0].axhline(0, linestyle="--", color="k")
ax[0].axvline(0, linestyle="--", color="k")

ax[1].scatter(mg["delta_norm"], mg["sim1_norm"], c=mg["mean_pupil_range"], cmap="plasma")
ax[1].axhline(0, linestyle="--", color="k")
ax[1].axvline(0, linestyle="--", color="k")

ax[2].scatter(mg["delta_norm"], mg["sim2_norm"], c=mg["mean_pupil_range"], cmap="plasma")
ax[2].axhline(0, linestyle="--", color="k")
ax[2].axvline(0, linestyle="--", color="k")

ax[3].scatter(mg["delta_norm"], mg["sim3_norm"], c=mg["mean_pupil_range"], cmap="plasma")
ax[3].axhline(0, linestyle="--", color="k")
ax[3].axvline(0, linestyle="--", color="k")

ax[4].scatter(mg["delta_norm"], mg["sim4_norm"], c=mg["mean_pupil_range"], cmap="plasma")
ax[4].axhline(0, linestyle="--", color="k")
ax[4].axvline(0, linestyle="--", color="k")

ax[5].scatter(mg["delta_norm"], mg["sim5_norm"], c=mg["mean_pupil_range"], cmap="plasma")
ax[5].axhline(0, linestyle="--", color="k")
ax[5].axvline(0, linestyle="--", color="k")