"""
Global variables for this analysis
"""

ALL_SITES = ['BOL005c', 'BOL006b', 'bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
            'DRX006b.e1:64', 'DRX006b.e65:128',
            'DRX007a.e1:64', 'DRX007a.e65:128',
            'DRX008b.e1:64', 'DRX008b.e65:128',
            'CRD016d', 'CRD017c']

CPN_SITES = ['AMT020a', 'AMT021b', 'AMT026a', 'ARM005e', 'ARM029a', 'ARM031a',
       'ARM032a', 'ARM033a', 'CRD004a', 'CRD005b', 'CRD018d', 'CRD019b'] # (batch 331)

PEG_SITES = ['AMT028b', 'AMT029a', 'AMT031a', 'AMT032a', 'ARM018a', 'ARM019a', 'ARM021b', 'ARM022b']
HIGHR_PEG_SITES = ['ARM018a', 'ARM019a', 'ARM021b', 'ARM022b']

# BRT032, BRT033, and BRT037 + TAR009d?? Excluded because flat pupil, I think...
# subset in which cross-validation can be performed
HIGHR_SITES = ['BOL005c', 'BOL006b', 'TAR010c', 'TAR017b', 
            'bbl086b', 'DRX006b.e1:64', 'DRX006b.e65:128', 
            'DRX007a.e1:64', 'DRX007a.e65:128', 
            'DRX008b.e1:64', 'DRX008b.e65:128',
            'CRD016d', 'CRD017c']


# subset in which cross-validation not possible
LOWR_SITES = [s for s in ALL_SITES if s not in HIGHR_SITES]

# data cutoff settings (just for plotting -- all statisical tests use all the data)
DU_MAG_CUT = (2, 8)
NOISE_INTERFERENCE_CUT = (0.2, 1)