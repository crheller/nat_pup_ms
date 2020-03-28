import load_results as ld
import matplotlib.pyplot as plt
import plotting as cplt

nc = ld.load_noise_correlation('rsc')
mask = (nc['p_all'] < 1) & (nc['site']=='BRT026c')

nc_1 = ld.load_noise_correlation('rsc_fft0-0.25')[mask]
nc_2 = ld.load_noise_correlation('rsc_fft0.25-1')[mask]
nc_3 = ld.load_noise_correlation('rsc_fft0.5-3')[mask]
nc_4 = ld.load_noise_correlation('rsc_fft2-10')[mask]
nc_5 = ld.load_noise_correlation('rsc_fft10-50')[mask]

nc_pr_1 = ld.load_noise_correlation('rsc_pr_fft0-0.25')[mask]
nc_pr_2 = ld.load_noise_correlation('rsc_pr_fft0.25-1')[mask]
nc_pr_3 = ld.load_noise_correlation('rsc_pr_fft0.5-3')[mask]
nc_pr_4 = ld.load_noise_correlation('rsc_pr_fft2-10')[mask]
nc_pr_5 = ld.load_noise_correlation('rsc_pr_fft10-50')[mask]


f, ax = plt.subplots(1, 2, sharey=True)

ax[0].errorbar([0, 1, 2, 3, 4], [nc_1['all'].mean(), nc_2['all'].mean(), nc_3['all'].mean(), nc_4['all'].mean(), nc_5['all'].mean()],
            yerr=[nc_1['all'].sem(), nc_2['all'].sem(), nc_3['all'].sem(), nc_4['all'].sem(), nc_5['all'].sem()], 
            color='green')

ax[0].errorbar([0, 1, 2, 3, 4], [nc_pr_1['all'].mean(), nc_pr_2['all'].mean(), nc_pr_3['all'].mean(), nc_pr_4['all'].mean(), nc_pr_5['all'].mean()],
            yerr=[nc_pr_1['all'].sem(), nc_pr_2['all'].sem(), nc_pr_3['all'].sem(), nc_pr_4['all'].sem(), nc_pr_5['all'].sem()], 
            color='purple')


ax[1].errorbar([0, 1, 2, 3, 4], [nc_pr_1['bp'].mean(), nc_pr_2['bp'].mean(), nc_pr_3['bp'].mean(), nc_pr_4['bp'].mean(), nc_pr_5['bp'].mean()],
            yerr=[nc_pr_1['bp'].sem(), nc_pr_2['bp'].sem(), nc_pr_3['bp'].sem(), nc_pr_4['bp'].sem(), nc_pr_5['bp'].sem()], 
            color='firebrick')

ax[1].errorbar([0, 1, 2, 3, 4], [nc_pr_1['sp'].mean(), nc_pr_2['sp'].mean(), nc_pr_3['sp'].mean(), nc_pr_4['sp'].mean(), nc_pr_5['sp'].mean()],
            yerr=[nc_pr_1['sp'].sem(), nc_pr_2['sp'].sem(), nc_pr_3['sp'].sem(), nc_pr_4['sp'].sem(), nc_pr_5['sp'].sem()], 
            color='navy')

#ax[1].errorbar([0, 1, 2, 3, 4], [nc_1['bp'].mean(), nc_2['bp'].mean(), nc_3['bp'].mean(), nc_4['bp'].mean(), nc_5['bp'].mean()],
#            yerr=[nc_1['bp'].sem(), nc_2['bp'].sem(), nc_3['bp'].sem(), nc_4['bp'].sem(), nc_5['bp'].sem()], 
#            color='firebrick')

#ax[1].errorbar([0, 1, 2, 3, 4], [nc_1['sp'].mean(), nc_2['sp'].mean(), nc_3['sp'].mean(), nc_4['sp'].mean(), nc_5['sp'].mean()],
#            yerr=[nc_1['sp'].sem(), nc_2['sp'].sem(), nc_3['sp'].sem(), nc_4['sp'].sem(), nc_5['sp'].sem()], 
#            color='navy')

ax[1].set_aspect(cplt.get_square_asp(ax[1]))
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

f.tight_layout()

plt.show()
