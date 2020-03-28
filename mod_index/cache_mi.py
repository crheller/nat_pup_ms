import nems_lbhb.baphy as nb
from nems_lbhb.preprocessing import mask_high_repetion_stims
from nems.recording import Recording
import sys
sys.path.append('/auto/users/hellerc/code/projects/nat_pupil_ms_final/')
import load_results as ld
sys.path.append('/auto/users/hellerc/code/crh_tools/')
import preprocessing as preproc
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

sites = ['bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
        'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
        'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b']
pupil_regress=True
for site in sites:

    path = '/auto/users/hellerc/results/nat_pupil_ms/mod_index/'

    log.info('Computing mod index for site: {0}'.format(site))

    log.info("Saving results to: {}".format(path))

    batch = 289
    fs = 4

    log.info("Load recording")
    ops = {'batch': batch, 'siteid': site, 'rasterfs': fs, 'pupil': 1, 'rem': 1,
        'stim': 1}
    uri = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    rec['resp'] = rec['resp'].rasterize()
    rec['stim'] = rec['stim'].rasterize()
    rec = mask_high_repetion_stims(rec)
    rec = rec.apply_mask(reset_epochs=True)
    rec = rec.create_mask(True)

    if pupil_regress:
         rec = preproc.regress_state2(rec, state_sigs=['pupil'], regress=['pupil'])


    rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    rec = rec.apply_mask(reset_epochs=True)

    # Create pupil masks
    p_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
    rec_bp = preproc.create_pupil_mask(rec, **p_ops)
    p_ops['state'] = 'small'
    rec_sp = preproc.create_pupil_mask(rec, **p_ops)

    # balanced epochs
    epochs = preproc.get_pupil_balanced_epochs(rec, rec_sp, rec_bp)
    rec_bp = rec_bp.and_mask(epochs)
    rec_sp = rec_sp.and_mask(epochs)
    rec_bp = rec_bp.apply_mask(reset_epochs=True)
    rec_sp = rec_sp.apply_mask(reset_epochs=True)

    # Compute coarse MI for each neuron, over all stimuli (balanced)
    Rbig = rec_bp['resp'].extract_epoch('REFERENCE').mean(0).mean(-1)
    Rsmall = rec_sp['resp'].extract_epoch('REFERENCE').mean(0).mean(-1)

    MI = (Rbig - Rsmall) / (Rbig + Rsmall)

    df = pd.DataFrame(index=rec['resp'].chans, data=MI, columns=['MI'])

    if pupil_regress:
        df.to_csv(path+site+'_pr_MI.csv')
    else:
        df.to_csv(path+site+'_MI.csv')
