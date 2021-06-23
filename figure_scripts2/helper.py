"""
temp file - WIP: a helper function for building cofusion matrices at a site
"""
import pandas as pd
import numpy as np
import charlieTools.plotting as cplt
import scipy.cluster.hierarchy as hc


def plot_confusion_matrix(df, metric, spectrogram, sortby=None, sort_method='full', 
                                        resp_fs=None, stim_fs=None, pcs=None, baseline=0,
                                        cmap='bwr', midpoint=0, vmin=None, vmax=None, pc_div=16, spec_div=16, ax=None):
    """
    df: pairwise decoding results
    metric: value to plot on the matrix
        column name existing in df
    spectrogram: freq bins X time 
        extent of heatmap will be forced to len(time bins)

    If pcs not None, also plot them (under the top spectrogram)
    Their sampling rate must be same as df (stim bins)
    pcs is a matrix of stim x trials x dims

    sortby - tuple: (df key, binsize (bins to bin the cfm by in order to sort))
    """
    if (resp_fs is None) | (stim_fs is None):
        raise ValueError

    if metric not in df.keys():
        raise ValueError

    # get matrix max based on length of spectrogram
    extent = int((spectrogram.shape[-1] / stim_fs) * resp_fs)

    n_components = df.index.get_level_values(1)[0]

    # fill confusion matrix
    
    cfm = np.nan * np.ones((extent, extent))
    for c in df.index.get_level_values(0):
        r = df.loc[pd.IndexSlice[c, n_components], :]
        c1 = int(c.split('_')[0])
        c2 = int(c.split('_')[1])
        cfm[c1, c2] = r[metric]
        cfm[c2, c1] = r[metric]
    
    if sortby is not None:

        cfm_sort = np.nan * np.ones((extent, extent))
        for c in df.index.get_level_values(0):
            r = df.loc[pd.IndexSlice[c, n_components], :]
            c1 = int(c.split('_')[0])
            c2 = int(c.split('_')[1])
            cfm_sort[c1, c2] = r[sortby[0]]
            cfm_sort[c2, c1] = r[sortby[0]]

        # reduce for clustering - slide window nanmean over the data
        bins = int(cfm.shape[0] / sortby[1])
        dscfm = np.zeros((bins, bins))
        for i in range(bins):
            for j in range(bins):
                if i != j:
                    xr = np.arange(i*sortby[1], (i*sortby[1]) + sortby[1])
                    yr = np.arange(j*sortby[1], (j*sortby[1]) + sortby[1])
                    val = np.nanmean(cfm_sort[xr, yr])
                    dscfm[i, j] = val

       
        # sort at level of full sound chunks, so that we don't totally jumble
        # the spectrograms
        if sort_method=='1D':
            o1 = np.argsort(dscfm.mean(axis=-1)).squeeze()
        elif sort_method=='full':
            # now, cluster this reduced matrix and sort stimuli based on this
            link = hc.linkage(dscfm, method='median', metric='euclidean')
            o1 = hc.leaves_list(link)
        cfm_ordered = np.zeros(cfm.shape)
        for i, _o1 in enumerate(o1):
            for j, _o2 in enumerate(o1):
                xr = np.arange(_o1*sortby[1], (_o1*sortby[1]) + sortby[1])
                yr = np.arange(_o2*sortby[1], (_o2*sortby[1]) + sortby[1])
                idx1 = 0
                for _i in xr:
                    idx2 = 0
                    for _j in yr:
                        val = cfm[_i, _j]
                        cfm_ordered[int(i*sortby[1])+idx1, int(j*sortby[1])+idx2] = val
                        cfm_ordered[int(j*sortby[1])+idx2, int(i*sortby[1])+idx1] = val
                        idx2 += 1
                    idx1 += 1 

        cfm = cfm_ordered

        # reorder spectrgram
        spec_new = np.zeros(spectrogram.shape)
        sbin = int((sortby[1]/resp_fs)*stim_fs)
        for i, o in enumerate(o1):
            idx = np.arange(int(i * sbin), int(i * sbin) + sbin)
            idxt = np.arange(int(o * sbin), int(o * sbin) + sbin)
            spec_new[:, idx] = spectrogram[:, idxt]
        
        spectrogram = spec_new

        # reorder PCs
        if pcs is not None:
            pcs_new = np.zeros(pcs.shape)
            for i, o in enumerate(o1):
                idx = np.arange(int(i * sortby[1]), int(i * sortby[1]) + sortby[1])
                idxt = np.arange(int(o * sortby[1]), int(o * sortby[1]) + sortby[1])
                pcs_new[idx, :, :] = pcs[idxt, :, :]
            pcs = pcs_new

    # layout of elements on a single axis
    if ax is None:
        f, ax = plt.subplots(1, 1)
    
    # add raster plot / pc response to plot

    # figure out extent of different panels
    spChan = extent / spec_div   # make spec height 1/8 of matrix 
    pcHeight = extent / pc_div
    #rastHeight = extent / 8

    # plot confusion matrix
    cfmflip = cfm.copy()
    for i in range(cfm.shape[0]):
        for j in range(cfm.shape[0]):
            cfmflip[i, j] = cfm[j, i]
    # upper triangle
    mappable = ax.imshow(cfmflip, extent=[0, extent, 0, extent], 
                        origin='lower', cmap=cmap, norm=cplt.MidpointNormalize(midpoint=midpoint, vmin=vmin, vmax=vmax))
    # lower triangle
    ax.imshow(cfm, extent=[0, extent, 0, extent], 
                        origin='lower', cmap=cmap, norm=cplt.MidpointNormalize(midpoint=midpoint, vmin=vmin, vmax=vmax))
    
    
    # Plot PC response / rasters
    #ax.imshow(raster, extent=[0, extent, extent+pcHeight, extent+pcHeight+rastHeight], cmap='plasma')

    # pcs -- these should *come* in scaled between 0 and 1, 
    # so, just scale by the pcHeight param - this way, all plots (i.e. big/small)
    # use the same scale
    if pcs is not None:
        # plot spectrograms
        # right spec (same as above)
        ax.imshow(np.fliplr(np.flipud(spectrogram).T), extent=[extent+pcHeight, spChan+extent+pcHeight, 0, extent], origin='lower', cmap='Greys')
        # top spec (need to offset for response)
        ax.imshow(spectrogram, extent=[0, extent, extent+pcHeight, 
                                            spChan+extent+pcHeight], origin='lower', cmap='Greys')
        t = np.linspace(0, extent, extent)
        pcsm = pcs.mean(axis=1)
        pcsm *= pcHeight
        pcsm += (extent)
        for i in range(pcs.shape[-1]):
            ax.plot(t, pcsm[:, i], lw=0.6)
            ax.plot(pcsm[:, i], t, lw=0.6, color=ax.get_lines()[-1].get_color())
        base = (baseline * pcHeight) + extent
        ax.plot(t, np.ones(t.shape)*base, lw=0.6, linestyle='--', color='grey', zorder=-1)
        ax.plot(np.ones(t.shape)*base, t, lw=0.6, linestyle='--', color='grey', zorder=-1)
    else:
        # put the spec where the PCs would've been (make it hug matrix)
        ax.imshow(np.fliplr(np.flipud(spectrogram).T), extent=[extent, spChan+extent, 0, extent], origin='lower', cmap='Greys')
        # top spec (need to offset for response)
        ax.imshow(spectrogram, extent=[0, extent, extent, 
                                            spChan+extent], origin='lower', cmap='Greys')

    ax.set_xlim((0, extent+spChan+pcHeight))
    ax.set_ylim((0, extent+spChan+pcHeight))
    
    ax.axis('off')

    #[ax.spines[x].set_visible(False) for x in ax.spines]

    return mappable