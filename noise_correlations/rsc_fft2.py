import load_results as ld
import matplotlib.pyplot as plt
import scipy.stats as ss

nc = ld.load_noise_correlation('rsc')
mask = (nc['p_all'] < 1) #& (nc['site']=='BRT026c')
rsc_path = '/auto/users/hellerc/results/nat_pupil_ms/noise_correlations/'

nc_1 = ld.load_noise_correlation('rsc_fft0.5-3')