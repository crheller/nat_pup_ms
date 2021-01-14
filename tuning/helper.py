"""
temp file - WIP: a helper function for building cofusion matrices at a site
"""
import pandas as pd
import numpy as np
import charlieTools.plotting as cplt

def plot_confusion_matrix(df, metric, spectrogram, resp_fs=None, stim_fs=None, cmap='bwr', midpoint=0, vmin=None, vmax=None, ax=None):
    """
    df: pairwise decoding results
    metric: value to plot on the matrix
        column name existing in df
    spectrogram: freq bins X time bins
        extent of heatmap will be forced to len(time bins)
    """
    if (resp_fs is None) | (stim_fs is None):
        raise ValueError

    if metric not in df.keys():
        raise ValueError

    # get matrix max based on length of spectrogram
    extent = int((spectrogram.shape[-1] / stim_fs) * resp_fs)

    # fill confusion matrix
    cfm = np.nan * np.ones((extent, extent))
    for c in df.index.get_level_values(0):
        r = df.loc[pd.IndexSlice[c, 2], :]
        c1 = int(c.split('_')[0])
        c2 = int(c.split('_')[1])
        cfm[c1, c2] = r[metric]

    # layout of elements on a single axis
    if ax is None:
        f, ax = plt.subplots(1, 1)
    
    spax = np.linspace(0, extent, spectrogram.shape[-1])
    spChan = spectrogram.shape[0]
    # plot confusion matrix
    ax.imshow(cfm, extent=[0, extent, spChan, extent+spChan], 
                        origin='lower', cmap=cmap, norm=cplt.MidpointNormalize(midpoint=midpoint, vmin=vmin, vmax=vmax))
    # plot spectrograms
    ax.imshow(np.flipud(spectrogram).T, extent=[extent, spChan+extent, spChan, extent+spChan], origin='lower', cmap='Greys')
    ax.imshow(spectrogram, extent=[0, extent, 0, spChan], origin='lower', cmap='Greys')

    #ax.imshow(np.nan * cfm, extent=[0, extent+spChan, 0, extent+spChan])
    ax.set_xlim((0, extent+spChan))
    ax.set_ylim((0, extent+spChan))

    ax.axis('off')

    #[ax.spines[x].set_visible(False) for x in ax.spines]