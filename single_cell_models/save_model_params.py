"""
Does this script already 
exist somewhere else??? CRH 01.23.2021

Adding it now for the PEG data...

Apparently, need to cache these results for the cache_LV script to run...
"""
from single_cell_models.mod_per_state import get_model_results_per_state_model
import os

batch = 331
d = get_model_results_per_state_model(batch=batch, state_list=['st.pup', 'st.pup0'], 
                                                  loader='ns.fs4.pup-ld-', 
                                                  fitter='_jk.nf10-basic',
                                                  basemodel='-hrc-psthfr_sdexp.SxR.bound')
path = '/auto/users/hellerc/results/nat_pupil_ms/first_order_model_results/'
d.to_csv(os.path.join(path, f'd_{batch}_pup_sdexp.csv'))