"""
New method of measuring dprime. Using loocv.
Idea is that this should make things more stable, by using more data.

09.23.2020

Procedure:
    1) Load data
    2) Generate list of stimulus pairs
    3) Make pupil mask, classify pupil for each stim pair
    4) Preprocess data (e.g. z-score)
    --- For each stim pair, loop over data sets of loocv ---
        5) Dimensionality reduction
        6) Compute dprime, save metrics
        7) Classify projected "left out" data points as large / small pupil, 
                    based on pupil mask generated at the beginning. Use these to compute state-dependent d'
        8) For state-dependent evecs/evals/dU, split train set using pupil mask, then take mean over all test sets

"""