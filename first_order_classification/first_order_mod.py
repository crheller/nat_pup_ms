"""
quanitfy / cache a summary of first order pupil mod for each site
so, load model fits and compare shuffled vs. unshuffled. Use jackknifed results
"""
from nems.xform_helper import load_model_xform
from nems_lbhb.baphy_io import parse_cellid
from global_settings import CPN_SITES, HIGHR_SITES

m = 'psth.fs4.pup-ld-st.pup-epcpn-hrc-psthfr.z_stategain.SxR_jk.nf10-tfinit.n.lr1e4.cont.et5.i50000'
m0 = 'psth.fs4.pup-ld-st.pup0-epcpn-hrc-psthfr.z_stategain.SxR_jk.nf10-tfinit.n.lr1e4.cont.et5.i50000'

sites = CPN_SITES + HIGHR_SITES
batches = [331] * len(CPN_SITES) + [322] * len(HIGHR_SITES)

for i, (site, batch) in enumerate(zip(sites, batches)):
    if site in ["BOL005c", "BOL006b"]:
        batch = 294
    if batch != 331:
        _m = m.replace("-epcpn", "")
        _m0 = m.replace("-epcpn", "")
    cellist, options = parse_cellid({"cellid": site, "batch": batch})
    xf, ctx = load_model_xform(cellid=cellist[0], batch=batch, modelname=_m)