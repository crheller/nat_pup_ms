"""
Global variables for this analysis
"""

ALL_SITES = ['BOL005c', 'BOL006b', 'bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
            'DRX006b.e1:64', 'DRX006b.e65:128',
            'DRX007a.e1:64', 'DRX007a.e65:128',
            'DRX008b.e1:64', 'DRX008b.e65:128']


# BRT032, BRT033, and BRT037 + TAR009d?? Excluded because flat pupil, I think...
# subset in which cross-validation can be performed
HIGHR_SITES = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
            'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
            'DRX007a.e1:64', 'DRX007a.e65:128', 
            'DRX008b.e1:64', 'DRX008b.e65:128']


# subset in which cross-validation not possible
LOWR_SITES = [s for s in ALL_SITES if s not in HIGHR_SITES]

# data cutoff settings (just for plotting -- all statisical tests use all the data)
DU_MAG_CUT = (2, 8)
NOISE_INTERFERENCE_CUT = (0.2, 1)