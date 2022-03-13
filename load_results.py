import pandas as pd
import os
import numpy as np
import sys

def load_noise_correlation(modelname, xforms_model='NULL', path=None, batch=None):

    if path is None:
        path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations_final/'
    else:
        pass

    if xforms_model is None:
        xforms_model = 'ns.fs4.pup-ld.pop-st.pup-hrc-apm-psthfr-addmeta-aev_stategain.SxR-lv.1xR.f.pred-stategain.2xR.lv_init.i0.xx1.t7-init.f0.t7.pLV0'
    else:
        pass

    if batch is None:
        batch = [322, 294, 331]
    elif type(batch) != list:
        batch = [batch]
    else:
        pass
    
    dfs = []
    for bat in batch:

        sites = os.listdir(os.path.join(path, str(bat)))

        for s in sites:
            if s in ['BOL005c', 'BOL006b']:
                xf_model = xforms_model.replace('fs4.pup', 'fs4.pup.voc')
            elif bat == 331:
                xf_model = xforms_model.replace('-hrc', '-epcpn-hrc')
            else:
                xf_model = xforms_model
            try:
                if ('fft' in modelname) | (xforms_model=='NULL'):
                    df = pd.read_csv(path+str(bat)+'/'+s+'/'+modelname+'.csv', index_col=[0])
                    df['batch'] = bat
                    dfs.append(df)               
                else:
                    df = pd.read_csv(path+str(bat)+'/'+s+'/'+xf_model+'/'+modelname+'.csv', index_col=0)
                    df['batch'] = bat
                    dfs.append(df)
            except:
                print("no results found for site: {0}, model: {1}".format(s, modelname))
    
    df = pd.concat(dfs)

    return df

def load_latent_variable(site):
    path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/GLM/latent_variables/'
    return np.load(path+site+'_latent_variable.npy')


def load_mi(pr=False):
    path = '/auto/users/results/projects/nat_pupil_ms/mod_index/'

    site_files = os.listdir(path)
    if pr:
        site_files = [s for s in site_files if 'pr' in s]
    else:
        site_files = [s for s in site_files if 'pr' not in s]
    
    sites = np.unique([s.split('_')[0] for s in site_files]).tolist()
    dfs = []
    for i, sf in enumerate(site_files):
        df = pd.read_csv(path+sf, index_col=0)
        df['site'] = sf.split('_')[0]
        dfs.append(df)
    return pd.concat(dfs)


def ddprime_alpha(site=None):
    
    path = '/auto/users/hellerc/code/projects/nat_pupil_ms_final/dprime/alphas/'
    if site is not None:
        return pd.read_csv(path+site+'_alpha.csv', index_col=0)
    else:
        files = os.listdir(path)
        dfs = []
        for f in files:
            d = pd.read_csv(path+f, index_col=0)
            d = d.sort_values(by=['slow', 'fast'])
            dfs.append(d)
        return pd.concat(dfs)
