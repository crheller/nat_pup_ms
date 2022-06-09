import nems.db as nd
import pandas as pd
import nems_lbhb.projects.nat_pup_decoding.decoding as decoding 
import os

def load_delta_dprime(site, batch, modelnames=[], columns=[]):
    """
    Return single df, merged on epoch name, where each value is delta dprime 
    between lrg small pupil for a different normative pop model.
    Columns labeled according to "columns"
    Returns two dfs -- one is raw delta dprime, one is norm. delta dprime
    """
    cellid = [c for c in nd.get_batch_cells(batch).cellid if site in c][0]
    loader = decoding.DecodingResults()
    norm_delta = []
    raw_delta = []
    for m, k in zip(modelnames, columns):
        modelpath = nd.get_results_file(batch=batch, modelnames=[m], cellids=[cellid]).iloc[0]["modelpath"]
        res = loader.load_results(os.path.join(modelpath, "decoding_TDR.pickle"))
        df = res.numeric_results
        df = df.loc[pd.IndexSlice[res.evoked_stimulus_pairs, 2], :].copy()
        
        df.loc[:, "raw_delta"] = df["bp_dp"] - df["sp_dp"]
        df.loc[:, "norm_delta"] = df["raw_delta"] / (df["bp_dp"] + df["sp_dp"])

        norm_delta.append(df[["norm_delta"]])
        raw_delta.append(df[["raw_delta"]])
    
    raw_df = pd.concat(raw_delta, axis=1)
    raw_df.columns = columns
    raw_df.loc[:, "site"] = site
    raw_df.loc[:, "batch"] = batch
    norm_df = pd.concat(norm_delta, axis=1)
    norm_df.columns = columns
    norm_df.loc[:, "site"] = site
    norm_df.loc[:, "batch"] = batch
    return raw_df, norm_df