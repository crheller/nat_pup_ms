"""
Noticed that movement / blinking leads to either bursts of correlated activity (or artificact, hard to say)
that then messes up PC / correlation computation.

Want to exclude these using new pupil analysis
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.preprocessing import fix_cpn_epochs, create_pupil_mask
from nems.preprocessing import generate_psth_from_resp
import matplotlib.pyplot as plt
import numpy as np
from charlieTools.noise_correlations import compute_rsc

manager = BAPHYExperiment(cellid='ARM029a', batch=331)
manager = BAPHYExperiment(cellid='ARM033a', batch=331)
manager = BAPHYExperiment(cellid='AMT026a', batch=331)
manager = BAPHYExperiment(cellid='CRD018d', batch=331)
manager = BAPHYExperiment(cellid='AMT020a', batch=331)
manager = BAPHYExperiment(cellid='ARM031a', batch=331)
r = manager.get_recording(recache=True, **{'rasterfs': 4, 'resp': True, 'pupil': True, 'stim': False, 
                                            'pupil_variable_name': 'area',
                                            'pupil_offset': 0.25,
                                            'pupil_deblink_dur': 1})      
r['resp'] = r['resp'].rasterize() 
r = fix_cpn_epochs(r)
r = generate_psth_from_resp(r)

epochs = [e for e in r['resp'].epochs.name.unique() if e.startswith('STIM')]
r = r.and_mask(epochs)

binsizes = [0.5, 1, 2]#, 3, 4]
thresh = 1
significant_pairs = False
for binsize in binsizes:
    rec = r.copy()
    # before masking, take raw signals and compute variance of an eyelid over a sliding window
    binsize = int(binsize * rec['resp'].fs)
    signal = rec['pupil_extras'].extract_channels(['eyelid_top_y'])._data 
    signal2 = rec['pupil_extras'].extract_channels(['eyelid_bottom_y'])._data
    varsig = np.zeros(signal.shape)
    for i in range(varsig.shape[-1]):
        varsig[0, i] = np.var(signal[0, i:(i+binsize)])+np.var(signal2[0, i:(i+binsize)])
    varsig = np.roll(varsig, int(binsize/2))
    rec['varsig'] = rec['pupil']._modified_copy(varsig)
    threshold = np.std(varsig)/4
    mask = varsig >= threshold

    # plot the raw data with the threshold
    f, ax = plt.subplots(4, 1, figsize=(10, 10))

    ax[0].set_title("Pupil")
    ax[0].plot(rec['pupil']._data.T)
    ax[0].set_xlim((0, rec['resp'].shape[-1]))

    ax[1].set_title("Eyespeed")
    ax[1].plot(rec['pupil_extras'].extract_channels(['eyespeed'])._data.T)
    ax[1].set_xlim((0, rec['resp'].shape[-1]))

    ax[2].set_title("Eyelid keypoints")
    d1 = rec['pupil_extras'].extract_channels(['eyelid_top_y'])._data.T
    ax[2].plot(d1/d1.max(), label='bottom')
    d2 = rec['pupil_extras'].extract_channels(['eyelid_bottom_y'])._data.T
    ax[2].plot(d2/d2.max()-0.5, label='top')
    d3 = rec['varsig']._data.T
    ax[2].plot(d3/d3.max(), label='var(bottom)')
    ax[2].axhline(threshold/d3.max(), linestyle='--', color='red', label='threshold')
    ax[2].legend(frameon=False)
    ax[2].set_xlim((0, rec['resp'].shape[-1]))

    ax[3].set_title("Spike raster")
    ax[3].imshow(rec['resp']._data-rec['psth_sp']._data, aspect='auto', cmap='Greys')

    f.tight_layout()

    # mask data and plot before / after noise correlations
    rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    bp = create_pupil_mask(rec, **{'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True})
    sp = create_pupil_mask(rec, **{'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True})
    bp = compute_rsc(bp['resp'].extract_epochs(epochs, mask=bp['mask']))
    sp = compute_rsc(sp['resp'].extract_epochs(epochs, mask=sp['mask']))

    if significant_pairs:
        m = (bp['pval'] < 0.05) | (sp['pval'] < 0.05)
    else:
        m = [True] * bp.shape[0]
    before_big = bp['rsc'][m]
    before_small = sp['rsc'][m]

    rec['var_mask'] = rec['pupil']._modified_copy(mask)
    fm = rec['var_mask'].extract_epochs(epochs)
    # tile bool across full reference so not to split up epochs in weird ways
    fm = {k: np.concatenate([np.zeros(v[[i]].shape).astype(bool) if np.any(v[[i]]==True) else np.ones(v[[i]].shape).astype(bool) 
                        for i in range(v.shape[0])], axis=0) 
                        for k, v in fm.items()}
    rec['var_mask'] = rec['var_mask'].replace_epochs(fm)
    rec['mask'] = rec['mask']._modified_copy(rec['mask']._data & rec['var_mask']._data)
    
    bp = create_pupil_mask(rec, **{'state': 'big', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True})
    sp = create_pupil_mask(rec, **{'state': 'small', 'method': 'median', 'epoch': ['REFERENCE'], 'collapse': True})
    bp = compute_rsc(bp['resp'].extract_epochs(epochs, mask=bp['mask']))
    sp = compute_rsc(sp['resp'].extract_epochs(epochs, mask=sp['mask']))

    after_big = bp['rsc'][m]
    after_small = sp['rsc'][m]

    f, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].set_title("All data")
    ax[0].scatter(before_big, before_small, s=5)
    ax[0].plot([-.5, .5], [-.5, .5], 'k--')
    ax[0].axhline(0, linestyle='--', color='k')
    ax[0].axvline(0, linestyle='--', color='k')
    ax[0].set_xlabel(f'Big: {round(before_big.mean(), 3)}')
    ax[0].set_ylabel(f'Small: {round(before_small.mean(), 3)}')

    ax[1].set_title(f"Mask movement with binsize: {binsize/r['resp'].fs} sec")
    ax[1].scatter(after_big, after_small, s=5)
    ax[1].plot([-.5, .5], [-.5, .5], 'k--')
    ax[1].axhline(0, linestyle='--', color='k')
    ax[1].axvline(0, linestyle='--', color='k')
    ax[1].set_xlabel(f'Big: {round(after_big.mean(), 3)}')
    ax[1].set_ylabel(f'Small: {round(after_small.mean(), 3)}')

    f.tight_layout()

    plt.show()