"""
quanitfy / cache a summary of first order pupil mod for each site
so, load model results and compare shuffled vs. unshuffled. Use jackknifed results.

For each site:
    * fraction of cells with significant pupil effect
    * mean magnitude of pupil effect -- mean r_test - r_test0
        * for just sig cells and for all cells
    * mean r_test0 (do sites with crummy auditory responses cause us problems?)
"""
from nems.xform_helper import load_model_xform
import nems.db as nd
from nems_lbhb.baphy_io import parse_cellid
from global_settings import CPN_SITES, HIGHR_SITES

import pandas as pd

saveto = "/auto/users/hellerc/code/projects/nat_pupil_ms/first_order_classification/first_order_mod.pickle"

m = 'psth.fs4.pup-ld-st.pup-epcpn-hrc-psthfr.z_stategain.SxR_jk.nf20-tfinit.n.lr1e4.cont.et5.i50000'
m0 = 'psth.fs4.pup-ld-st.pup0-epcpn-hrc-psthfr.z_stategain.SxR_jk.nf20-tfinit.n.lr1e4.cont.et5.i50000'

sites = CPN_SITES + HIGHR_SITES
batches = [331] * len(CPN_SITES) + [322] * len(HIGHR_SITES)
results = pd.DataFrame(index=range(len(sites)), columns=['site', 'batch', 'nCells', 'nSig', 'rdiff_mag', 'rdiff_mag_sig', 'r_test0_mean'])
for idx, (site, batch) in enumerate(zip(sites, batches)):
    print(f"site: {site}, batch: {batch}")
    if site in ["BOL005c", "BOL006b"]:
        batch = 294
    if batch != 331:
        _m = m.replace("-epcpn", "")
        _m0 = m0.replace("-epcpn", "")
    else:
        _m = m
        _m0 = m0
    cellist, options = parse_cellid({"cellid": site, "batch": batch})
    d = nd.pd_query(f"SELECT cellid, r_test, se_test FROM Results WHERE modelname='{_m}' and cellid in {tuple(cellist)}")
    d0 = nd.pd_query(f"SELECT cellid, r_test, se_test FROM Results WHERE modelname='{_m0}' and cellid in {tuple(cellist)}")

    nCells = len(cellist)
    sigmask = (d["r_test"]-d0["r_test"]) > (d["se_test"] + d0["se_test"])
    nSig = sum(sigmask)
    rdiff_mag = (d["r_test"] - d0["r_test"]).mean()
    rdiff_mag_sig = (d["r_test"][sigmask] - d0["r_test"][sigmask]).mean()
    r_test0_mean = d0["r_test"].mean()
    results.iloc[idx, :] = [site, batch, nCells, nSig, rdiff_mag, rdiff_mag_sig, r_test0_mean]


results.to_pickle(saveto)