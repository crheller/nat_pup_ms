"""
temp file - WIP: a helper function for building cofusion matrices at a site
"""
import pandas as pd
import numpy as np
import charlieTools.plotting as cplt

def plot_confusion_matrix(df, metric, spectrogram, resp_fs=None, stim_fs=None, 
                                        pcs=None,
                                        cmap='bwr', midpoint=0, vmin=None, vmax=None, ax=None):
    """
    df: pairwise decoding results
    metric: value to plot on the matrix
        column name existing in df
    spectrogram: freq bins X time bins
        extent of heatmap will be forced to len(time bins)

    If pcs not None, also plot them (under the top spectrogram)
    Their sampling rate must be same as df (stim bins)
    pcs is a matrix of stim x trials x dims
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
    
    if (pcs is None):
        spChan = extent / 8 # make spec height 1/8 of matrix # spectrogram.shape[0]
        # plot confusion matrix
        cfmflip = cfm.copy()
        for i in range(cfm.shape[0]):
            for j in range(cfm.shape[0]):
                cfmflip[i, j] = cfm[j, i]
        ax.imshow(cfmflip, extent=[0, extent, 0, extent], 
                            origin='lower', cmap=cmap, norm=cplt.MidpointNormalize(midpoint=midpoint, vmin=vmin, vmax=vmax))
        ax.imshow(cfm, extent=[0, extent, 0, extent], 
                            origin='lower', cmap=cmap, norm=cplt.MidpointNormalize(midpoint=midpoint, vmin=vmin, vmax=vmax))
        # plot spectrograms
        # right spec
        ax.imshow(np.fliplr(np.flipud(spectrogram).T), extent=[extent, spChan+extent, 0, extent], origin='lower', cmap='Greys')
        # top spec
        ax.imshow(spectrogram, extent=[0, extent, extent, spChan+extent], origin='lower', cmap='Greys')

        ax.set_xlim((0, extent+spChan))
        ax.set_ylim((0, extent+spChan))
    
    elif (pcs is not None):
        # add raster plot / pc response to plot

        # figure out extent of different panels
        spChan = extent / 16   # make spec height 1/8 of matrix 
        pcHeight = extent / 16 
        #rastHeight = extent / 8

        # plot confusion matrix
        cfmflip = cfm.copy()
        for i in range(cfm.shape[0]):
            for j in range(cfm.shape[0]):
                cfmflip[i, j] = cfm[j, i]
        # upper triangle
        ax.imshow(cfmflip, extent=[0, extent, 0, extent], 
                            origin='lower', cmap=cmap, norm=cplt.MidpointNormalize(midpoint=midpoint, vmin=vmin, vmax=vmax))
        # lower triangle
        ax.imshow(cfm, extent=[0, extent, 0, extent], 
                            origin='lower', cmap=cmap, norm=cplt.MidpointNormalize(midpoint=midpoint, vmin=vmin, vmax=vmax))
        
        
        # plot spectrograms
        # right spec (same as above)
        ax.imshow(np.fliplr(np.flipud(spectrogram).T), extent=[extent, spChan+extent, 0, extent], origin='lower', cmap='Greys')
        # top spec (need to offset for response)
        ax.imshow(spectrogram, extent=[0, extent, extent+pcHeight, 
                                                spChan+extent+pcHeight], origin='lower', cmap='Greys')


        # Plot PC response / rasters
        #ax.imshow(raster, extent=[0, extent, extent+pcHeight, extent+pcHeight+rastHeight], cmap='plasma')

        # pcs -- these should *come* in scaled between 0 and 1, 
        # so, just scale by the pcHeight param - this way, all plots (i.e. big/small)
        # use the same scale
        t = np.linspace(0, extent, extent)
        pcsm = pcs.mean(axis=1)
        pcsm *= pcHeight
        pcsm += (extent)
        for i in range(pcs.shape[-1]):
            ax.plot(t, pcsm[:, i], lw=0.8)

        ax.set_xlim((0, extent+spChan))
        ax.set_ylim((0, extent+spChan+pcHeight))
    ax.axis('off')

    #[ax.spines[x].set_visible(False) for x in ax.spines]