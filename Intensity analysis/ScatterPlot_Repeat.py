import numpy as np
from scipy import stats
from numpy import fft
import matplotlib.pyplot as plt
import gc
import pingouin as pg
import pandas as pd
# from scipy.stats import norm
# from scipy.optimize import curve_fit
# from tifffile import imsave
import pandas
import matplotlib
matplotlib.use("Qt5Agg")


# folderPath_hv = r"F:\Data_2024\20240626_jurkat\hv-0hr\3D particle analysis"
folderPath_lv = r"F:\Data_2024\20240626_jurkat\lv-0hr\3D particle analysis"
folderPath_mv = r"F:\Data_2024\20240626_jurkat\mv-0hr\3D particle analysis"
analyExcel = "Statistics for Data_3d_view" + ".csv"
# excelpath_hv = folderPath_hv + "\\" + analyExcel
excelpath_lv = folderPath_lv + "\\" + analyExcel
excelpath_mv = folderPath_mv + "\\" + analyExcel

# # # # # # # # # mean intensity # # # # # # # # #
# df = pandas.read_csv(excelpath_hv);  meanInt_list_hv = df['Mean'][:];  ptcolor = 'gray'
df = pandas.read_csv(excelpath_lv);  meanInt_list_lv = df['Mean'][:];  ptcolor = 'red'
df = pandas.read_csv(excelpath_mv);  meanInt_list_mv = df['Mean'][:];  ptcolor = 'green'
# # # diameter (width of bounding box (within bscan), not much trust on Y-scan)
# df = pandas.read_csv(excelpath_hv); dia_list_hv = (df['B-width'][:] + df['B-depth'][:]) / 2
df = pandas.read_csv(excelpath_lv);  dia_list_lv = (df['B-width'][:]) #+ df['B-depth'][:]) / 2
df = pandas.read_csv(excelpath_mv);  dia_list_mv = (df['B-width'][:]) #+ df['B-depth'][:]) / 2
# # # # # # # # # std of diameter (super linear to mean diameter, seems not useful) # # # # # #
# # df = pandas.read_csv(excelpath_hv);  ptcolor = 'gray'
# # if 'SD dist. to surf. (pixel)' in df.columns: circ_list_hv = df['SD dist. to surf. (pixel)'][:]
# # elif 'SD dist. to surf. ( )' in df.columns:  circ_list_hv = df['SD dist. to surf. ( )'][:]
# df = pandas.read_csv(excelpath_lv);  ptcolor = 'red'
# if 'Mean dist. to surf. (pixel)' in df.columns:  circ_list_lv = df['Mean dist. to surf. (pixel)'][:]
# elif 'Mean dist. to surf. ( )' in df.columns:  circ_list_lv = df['Mean dist. to surf. ( )'][:]
# df = pandas.read_csv(excelpath_mv);  ptcolor = 'green'
# if 'Mean dist. to surf. (pixel)' in df.columns:  circ_list_mv = df['Mean dist. to surf. (pixel)'][:]
# elif 'Mean dist. to surf. ( )' in df.columns:  circ_list_mv = df['Mean dist. to surf. ( )'][:]
# # # # # # # # # # # # x,z,y size (side length of bounding box) of object # # # # # # # # # # # #
# df = pandas.read_csv(excelpath_hv);  asp_ratio_hv = df['B-height'][:] / dia_list_hv
df = pandas.read_csv(excelpath_lv);  asp_ratio_lv = df['B-height'][:] / dia_list_lv
df = pandas.read_csv(excelpath_mv);  asp_ratio_mv = df['B-height'][:] / dia_list_mv

# # # repeatability via t-test #
# request_c = np.array(circ_list_lv)  # 对照组
# request_e = np.array(circ_list_hv)  # 实验组(真
# t, pval = stats.ttest_ind(request_e, request_c, equal_var=False); print(t); print(pval)
# # # repeatability via intra class correlation test #
# ratings = circ_list_hv + circ_list_lv + circ_list_mv
# raters = ['hv']*len(circ_list_hv) + ['lv']*len(circ_list_lv) + ['mv']*len(circ_list_mv)
# targets = # 因为无法确定测量的值是不是同一个细胞，所以ICC不适用
# data = pd.DataFrame({'targets':targets, 'raters':raters, 'ratings':ratings})
# icc = pg.intraclass_corr(data=data, targets='targets',
#                          raters='raters', ratings='ratings')
# print(icc.set_index('Type'))

# # - - - - - remove object with diameter < 10 um - - - -
# ind = [i for i in range(len(dia_list_lv)) if dia_list_lv[i] < 5]
# for index in sorted(ind, reverse=True): del dia_list_lv[index], asp_ratio_lv[index], meanInt_list_lv[index]

# - - - - - 3D scatter plotting func chuck - - - - - - - - - -
fig = plt.figure(11, figsize=(10, 10));  plt.clf()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('Diameter (um)');  ax.set_ylabel('Mean intensity (a.u.)'); ax.set_zlabel('Z-X aspect ratio (a.u.)')
# ax.scatter(dia_list_hv*2, meanInt_list_hv, circ_list_hv, color='gray')
ax.scatter(dia_list_lv*2, meanInt_list_lv, asp_ratio_lv, color='red', alpha=0.3, s=4)
ax.scatter(dia_list_mv*2, meanInt_list_mv, asp_ratio_mv, color='green', alpha=0.3, s=4)

ax.set_xlim3d(0, 50)  # Diameter
ax.set_ylim3d(30, 80)  # Mean intensity
ax.set_zlim3d(0, 4)  # Aspect ratio
ax.view_init(elev=31, azim=23, roll=0)   # (elev=10, azim=59, roll=0)


