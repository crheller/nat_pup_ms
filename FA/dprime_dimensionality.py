"""
How many FA dimensions do we need for each site before decoding prediction quality plateaus?
"""
from loader import load_pop_metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 12

import sys
sys.path.append("/auto/users/hellerc/code/projects/nat_pupil_ms/")
from global_settings import CPN_SITES, HIGHR_SITES
# decoding stuff
import charlieTools.nat_sounds_ms.decoding as decoding
from path_settings import DPRIME_DIR
import os

sites = CPN_SITES + HIGHR_SITES
batches = [331]*len(CPN_SITES) + [322 if s.startswith("BOL")==False else 294 for s in HIGHR_SITES]

noise = ""
noise331 = "_noiseDim-6"
loader = decoding.DecodingResults()

# load results
raw_df = []
fa_null_df = []
fa_ind_df = []
fa_full_df = []
fa_rr1_df = []
fa_rr2_df = []
fa_rr3_df = []
fa_rr4_df = []
fa_rr5_df = []
for i, (s, b) in enumerate(zip(sites, batches)):
    if b == 331:
        _noise = noise331
    else:
        _noise = noise
    rmodel = f"dprime_jk10_zscore_fixtdr2-fa{_noise}"                    # reaw data
    famodel_null = f"dprime_faModel.ind-null_jk10_zscore_fixtdr2-fa{_noise}" # fixed cov matrix between lrg / small
    famodel_ind = f"dprime_faModel.ind_jk10_zscore_fixtdr2-fa{_noise}"   # only change ind. variance. (diag cov matrix)
    famodel = f"dprime_faModel_jk10_zscore_fixtdr2-fa{_noise}"
    famodel_rr1 = f"dprime_faModel.rr1_jk10_zscore_fixtdr2-fa{_noise}"
    famodel_rr2 = f"dprime_faModel.rr2_jk10_zscore_fixtdr2-fa{_noise}"
    famodel_rr3 = f"dprime_faModel.rr3_jk10_zscore_fixtdr2-fa{_noise}"
    famodel_rr4 = f"dprime_faModel.rr4_jk10_zscore_fixtdr2-fa{_noise}"  
    famodel_rr5 = f"dprime_faModel.rr5_jk10_zscore_fixtdr2-fa{_noise}"

    fn = os.path.join(DPRIME_DIR, str(b), s, rmodel+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=None, recache=False)
    df = results.numeric_results.loc[results.evoked_stimulus_pairs]

    fn = os.path.join(DPRIME_DIR, str(b), s, famodel_null+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=None, recache=False)
    df_null_fa = results.numeric_results.loc[results.evoked_stimulus_pairs]

    fn = os.path.join(DPRIME_DIR, str(b), s, famodel_ind+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=None, recache=False)
    df_ind_fa = results.numeric_results.loc[results.evoked_stimulus_pairs]

    fn = os.path.join(DPRIME_DIR, str(b), s, famodel+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=None, recache=False)
    df_fa = results.numeric_results.loc[results.evoked_stimulus_pairs]

    # reduced rank models
    fn = os.path.join(DPRIME_DIR, str(b), s, famodel_rr1+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=None, recache=False)
    df_rr1 = results.numeric_results.loc[results.evoked_stimulus_pairs]

    fn = os.path.join(DPRIME_DIR, str(b), s, famodel_rr2+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=None, recache=False)
    df_rr2 = results.numeric_results.loc[results.evoked_stimulus_pairs]

    fn = os.path.join(DPRIME_DIR, str(b), s, famodel_rr3+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=None, recache=False)
    df_rr3 = results.numeric_results.loc[results.evoked_stimulus_pairs]

    fn = os.path.join(DPRIME_DIR, str(b), s, famodel_rr4+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=None, recache=False)
    df_rr4 = results.numeric_results.loc[results.evoked_stimulus_pairs]

    fn = os.path.join(DPRIME_DIR, str(b), s, famodel_rr5+'_TDR.pickle')
    results = loader.load_results(fn, cache_path=None, recache=False)
    df_rr5 = results.numeric_results.loc[results.evoked_stimulus_pairs]

    df["delta"] = (df["bp_dp"]-df["sp_dp"]) #/ (df["bp_dp"]+df["sp_dp"])
    df["site"] = s
    df["batch"] = b

    df_null_fa["delta"] = (df_null_fa["bp_dp"]-df_null_fa["sp_dp"]) #/(df_null_fa["bp_dp"]+df_null_fa["sp_dp"])
    df_null_fa["site"] = s
    df_null_fa["batch"] = b

    df_ind_fa["delta"] = (df_ind_fa["bp_dp"]-df_ind_fa["sp_dp"]) #/(df_ind_fa["bp_dp"]+df_ind_fa["sp_dp"])
    df_ind_fa["site"] = s
    df_ind_fa["batch"] = b

    df_fa["delta"] = (df_fa["bp_dp"]-df_fa["sp_dp"]) #/(df_fa["bp_dp"]+df_fa["sp_dp"])
    df_fa["site"] = s
    df_fa["batch"] = b

    # reduced rank models
    df_rr1["delta"] = (df_rr1["bp_dp"]-df_rr1["sp_dp"]) #/(df_rr1["bp_dp"]+df_rr1["sp_dp"])
    df_rr1["site"] = s
    df_rr1["batch"] = b

    df_rr2["delta"] = (df_rr2["bp_dp"]-df_rr2["sp_dp"]) #/(df_rr2["bp_dp"]+df_rr2["sp_dp"])
    df_rr2["site"] = s
    df_rr2["batch"] = b

    df_rr3["delta"] = (df_rr3["bp_dp"]-df_rr3["sp_dp"]) #/(df_rr3["bp_dp"]+df_rr3["sp_dp"])
    df_rr3["site"] = s
    df_rr3["batch"] = b

    df_rr4["delta"] = (df_rr4["bp_dp"]-df_rr4["sp_dp"]) #/(df_rr4["bp_dp"]+df_rr4["sp_dp"])
    df_rr4["site"] = s
    df_rr4["batch"] = b

    df_rr5["delta"] = (df_rr5["bp_dp"]-df_rr5["sp_dp"]) #/(df_rr5["bp_dp"]+df_rr5["sp_dp"])
    df_rr5["site"] = s
    df_rr5["batch"] = b

    raw_df.append(df)
    fa_null_df.append(df_null_fa)
    fa_ind_df.append(df_ind_fa)
    fa_full_df.append(df_fa)
    fa_rr1_df.append(df_rr1)
    fa_rr2_df.append(df_rr2)
    fa_rr3_df.append(df_rr3)
    fa_rr4_df.append(df_rr4)
    fa_rr5_df.append(df_rr5)

df = pd.concat(raw_df)
df_null_fa = pd.concat(fa_null_df)
df_ind_fa = pd.concat(fa_ind_df)
df_full_fa = pd.concat(fa_full_df)
df_rr1_fa = pd.concat(fa_rr1_df)
df_rr2_fa = pd.concat(fa_rr2_df)
df_rr3_fa = pd.concat(fa_rr3_df)
df_rr4_fa = pd.concat(fa_rr4_df)
df_rr5_fa = pd.concat(fa_rr5_df)

# make a "master" delta df
df["null_fa_delta"] = df_null_fa["delta"]
df["ind_fa_delta"] = df_ind_fa["delta"]
df["full_fa_delta"] = df_full_fa["delta"]
df["rr1_fa_delta"] = df_rr1_fa["delta"]
df["rr2_fa_delta"] = df_rr2_fa["delta"]
df["rr3_fa_delta"] = df_rr3_fa["delta"]
df["rr4_fa_delta"] = df_rr4_fa["delta"]
df["rr5_fa_delta"] = df_rr5_fa["delta"]

# compute abs error for each
df["null_err"] = abs(df["null_fa_delta"] - df["delta"])
df["ind_err"] = abs(df["ind_fa_delta"] - df["delta"])
df["full_err"] = abs(df["full_fa_delta"] - df["delta"])
df["rr1_err"] = abs(df["rr1_fa_delta"] - df["delta"])
df["rr2_err"] = abs(df["rr2_fa_delta"] - df["delta"])
df["rr3_err"] = abs(df["rr3_fa_delta"] - df["delta"])
df["rr4_err"] = abs(df["rr4_fa_delta"] - df["delta"])
df["rr5_err"] = abs(df["rr5_fa_delta"] - df["delta"])

# plot error per model per site / batch
err = df.groupby(by=["site", "batch"]).mean()[["null_err", "ind_err", "rr1_err", "rr2_err", "rr3_err", "rr4_err", "rr5_err", "full_err"]]
err_sem = df.groupby(by=["site", "batch"]).sem()[["null_err", "ind_err", "rr1_err", "rr2_err", "rr3_err", "rr4_err", "rr5_err", "full_err"]]
enan = []
f, ax = plt.subplots(8, 5, figsize=(15, 15))

for i in range(err.shape[0]):
    norm = err.iloc[i][0]
    e = err.iloc[i][:2] / norm #/ np.max(err.iloc[i])
    e_sem = err_sem.iloc[i][:2] / norm
    #e[setnan] = np.nan
    ax.flatten()[i].errorbar(range(e.shape[0]), e, e_sem, marker="o", color="r", alpha=0.5)

    dimmax = np.diff(err.iloc[i])
    #setnan = np.argwhere(abs(dimmax)==0)
    e = err.iloc[i][2:] / norm #/ np.max(err.iloc[i])
    e_sem = err_sem.iloc[i][2:] / norm
    #e[setnan] = np.nan
    ax.flatten()[i].errorbar(np.arange(0, e.shape[0])+2, e, e_sem, marker="o", color="grey", alpha=0.5)
    enan.append(e)
    ax.flatten()[i].set_title(err.index.values[i])
    ax.flatten()[i].axhline(1, linestyle="--", color="k")
f.tight_layout()