import pandas as pd 
import os
import numpy as np
import logging
log = logging.getLogger(__name__)


def load_dprime(modelname, xforms_model=None, path=None):

    if path is None:
        path = '/auto/users/hellerc/results/nat_pupil_ms/dprime/'

    if xforms_model is None:
        xforms_model = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0'

    sites = os.listdir(path)

    dfs = []
    for s in sites:
        if s in ['BOL005c', 'BOL006b']:
            xm = xforms_model.replace('fs4.pup', 'fs4.pup.voc')
        else:
            xm = xforms_model
        try:
            f = os.path.join(path, s, xm, modelname+'.csv')
            d = pd.read_csv(f, index_col=0)
            dfs.append(d)
        except:
            log.info("no results found for site: {}".format(s))

    df = pd.concat(dfs)

    for c in df.columns:
        if 'pc1' in c:
            df[c] = [float(x.replace('[', '').replace(']', '')) for x in df[c]]

    df['colFromIndex'] = df.index
    df = df.sort_values(by=['site', 'colFromIndex'])
    df = df.drop(columns='colFromIndex')

    return df


# ================ functions for masking dprime data frames ==============
# "global params"
hi_pc1 = 0.55
lo_similarity = 0.6
hi_similarity = 0.95
#lo_similarity = 0
#hi_similarity = 1.5
#hi_pc1 = 0.55

def filter_df(df, dp_all, dp_sp, site=None):
    """
    Remove extreme values, filter based on site (if site is not None)

    Purpose of function is just to have a common place where all this is 
    hardcoded.

    if df is a list, return the list of filtered dfs
    """
    xax = 'similarity'
    yax = 'pc1_proj_on_dec'
    X = dp_all[xax]
    Y = abs(dp_sp[yax])

    # define mask to get rid of outliers
    data_mask = (X > lo_similarity) & \
                (X < hi_similarity) & \
                (Y < hi_pc1) & \
                (dp_all['dprime'] > 0)  & \
                (dp_all['dprime'] < 15)
    if 'sound_sound' in dp_all['category'].values:
        data_mask = data_mask & \
                (dp_all['category'] == 'sound_sound')
    if site is not None:
        data_mask = data_mask & (dp_all['site'] == site)

    if type(df) is list:
        df_list = []
        for d in df:
            d_new = d.copy()
            df_list.append(d_new[data_mask])
        return df_list
    else:
        return df[data_mask]


def get_quad_mask(dp_all, dp_sp, quadrant):
    """
    Return mask for quad 1, 2, 3, or 4
    """
    xax = 'similarity'
    yax = 'pc1_proj_on_dec'
    #xax = 'difference'
    #yax = 'pc1_proj_on_dec_all'
    X = dp_all[xax]
    Y = abs(dp_sp[yax])

    Ydiv = np.median([0, hi_pc1])
    Xdiv = np.median([lo_similarity, hi_similarity])

    q1_mask = (X > Xdiv) & (Y > Ydiv)
    q2_mask = (X < Xdiv) & (Y > Ydiv)
    q3_mask = (X < Xdiv) & (Y < Ydiv)
    q4_mask = (X > Xdiv) & (Y < Ydiv)

    if quadrant == 1:
        return q1_mask
    elif quadrant == 2:
        return q2_mask
    elif quadrant == 3:
        return q3_mask
    elif quadrant == 4:
        return q4_mask


def quadrant_mask(df, dp_all, dp_sp, quadrant):
    """
    Return df masked for quad 1, 2, 3, or 4
    """
    xax = 'similarity'
    yax = 'pc1_proj_on_dec'
    X = dp_all[xax]
    Y = abs(dp_sp[yax])

    Ydiv = np.median([0, hi_pc1])
    Xdiv = np.median([lo_similarity, hi_similarity])

    q1_mask = (X > Xdiv) & (Y > Ydiv)
    q2_mask = (X < Xdiv) & (Y > Ydiv)
    q3_mask = (X < Xdiv) & (Y < Ydiv)
    q4_mask = (X > Xdiv) & (Y < Ydiv)

    if quadrant==1:
        return df[q1_mask]
    if quadrant==2:
        return df[q2_mask]
    if quadrant==3:
        return df[q3_mask]
    if quadrant==4:
        return df[q4_mask]
