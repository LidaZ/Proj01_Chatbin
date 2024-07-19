import numpy as np
from scipy import stats
from numpy import fft
import matplotlib.pyplot as plt
import gc
import pingouin as pg
import pandas as pd
from scipy.stats import pearsonr
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

# # - - - - - - - read parameters from .csv by "3D object counter" - - - - - - - - -
# # - - - - - - - macro available in \Fiji_macro\20240628_Auto3DJurkatAnaly.ijm - -
# # # mean intensity
# df = pandas.read_csv(excelpath_hv);  meanInt_list_hv = df['Mean'][:];  ptcolor = 'gray'
df = pandas.read_csv(excelpath_lv);  meanInt_list_lv = df['Mean'][:];  ptcolor = 'red'
df = pandas.read_csv(excelpath_mv);  meanInt_list_mv = df['Mean'][:];  ptcolor = 'green'
# # # diameter (width of bounding box (within bscan), not much trust on Y-scan)
# df = pandas.read_csv(excelpath_hv); dia_list_hv = (df['B-width'][:] + df['B-depth'][:]) / 2
df = pandas.read_csv(excelpath_lv);  dia_list_lv = (df['B-width'][:]) #+ df['B-depth'][:]) / 2
df = pandas.read_csv(excelpath_mv);  dia_list_mv = (df['B-width'][:]) #+ df['B-depth'][:]) / 2
# # # std of diameter (super linear to mean diameter, seems not useful)
# # df = pandas.read_csv(excelpath_hv);  ptcolor = 'gray'
# # if 'SD dist. to surf. (pixel)' in df.columns: circ_list_hv = df['SD dist. to surf. (pixel)'][:]
# # elif 'SD dist. to surf. ( )' in df.columns:  circ_list_hv = df['SD dist. to surf. ( )'][:]
# df = pandas.read_csv(excelpath_lv);  ptcolor = 'red'
# if 'Mean dist. to surf. (pixel)' in df.columns:  circ_list_lv = df['Mean dist. to surf. (pixel)'][:]
# elif 'Mean dist. to surf. ( )' in df.columns:  circ_list_lv = df['Mean dist. to surf. ( )'][:]
# df = pandas.read_csv(excelpath_mv);  ptcolor = 'green'
# if 'Mean dist. to surf. (pixel)' in df.columns:  circ_list_mv = df['Mean dist. to surf. (pixel)'][:]
# elif 'Mean dist. to surf. ( )' in df.columns:  circ_list_mv = df['Mean dist. to surf. ( )'][:]
# # # x,z,y size (side length of bounding box) of object
# df = pandas.read_csv(excelpath_hv);  asp_ratio_hv = df['B-height'][:] / dia_list_hv
df = pandas.read_csv(excelpath_lv);  asp_ratio_lv = df['B-height'][:] / dia_list_lv
df = pandas.read_csv(excelpath_mv);  asp_ratio_mv = df['B-height'][:] / dia_list_mv


# - - - - - - - - - Box plot: mean intensity vs. time - - - - - - - - - - -
if ('-0hr' in folderPath_lv) and ('-0hr' in folderPath_mv):
    total_meanInt_0hr_lv = meanInt_list_lv; total_meanInt_0hr_mv = meanInt_list_mv
if ('-1hr' in folderPath_lv) and ('-1hr' in folderPath_mv):
    total_meanInt_1hr_lv = meanInt_list_lv; total_meanInt_1hr_mv = meanInt_list_mv
if ('-2hr' in folderPath_lv) and ('-2hr' in folderPath_mv):
    total_meanInt_2hr_lv = meanInt_list_lv; total_meanInt_2hr_mv = meanInt_list_mv
if ('-3hr' in folderPath_lv) and ('-3hr' in folderPath_mv):
    total_meanInt_3hr_lv = meanInt_list_lv; total_meanInt_3hr_mv = meanInt_list_mv
if ('-4hr' in folderPath_lv) and ('-4hr' in folderPath_mv):
    total_meanInt_4hr_lv = meanInt_list_lv; total_meanInt_4hr_mv = meanInt_list_mv


# - - - - - Repeat above code to read through 0-hour to 3-hour, then execute following code for plotting
plt.figure(12); plt.clf();
data_lv = [total_meanInt_0hr_lv, total_meanInt_1hr_lv, total_meanInt_2hr_lv, total_meanInt_3hr_lv, total_meanInt_4hr_lv]
data_mv = [total_meanInt_0hr_mv, total_meanInt_1hr_mv, total_meanInt_2hr_mv, total_meanInt_3hr_mv, total_meanInt_4hr_mv]
labels = ['0-hour', '1-hour', '2-hour', '3-hour', '4-hour']
offsets_lv = np.arange(len(data_lv)) - 0.17
bp = plt.boxplot(data_lv, widths=0.3, positions=offsets_lv, patch_artist=True, sym='x')
colors = ['pink','pink','pink','pink','pink']
for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)

offsets_mv = np.arange(len(data_lv)) + 0.17
bp1 = plt.boxplot(data_mv, widths=0.3, positions=offsets_mv, patch_artist=True, sym='x');
colors = ['lightgreen','lightgreen','lightgreen','lightgreen','lightgreen']
for patch, color in zip(bp1['boxes'], colors): patch.set_facecolor(color)

plt.xticks([0, 1, 2, 3, 4], labels)


# - - - - Welch T-test to find out significant difference of mean intensity - - - -
request_c = np.array(total_meanInt_4hr_lv)  # 对照组
request_e = np.array(total_meanInt_4hr_mv)  # 实验组(真
_, pval = stats.ttest_ind(request_e, request_c, equal_var=False);  print(pval)


# - - - - Pearson test to find out correlation between viability, mean intensity, diameter, and aspect ratio - - - -
viability_oxy = [82.00, 59.10, 49.70, 45.20, 49.70]; viability_cont = [86.90, 83.60, 85.90, 84.90, 85.90]
viability = viability_oxy + viability_cont  # oxydol-treated and control
meanInt_oxy = [49.00,50.13,52.54,50.54,49.85]; meanInt_cont = [50.68,48.75,47.33,47.32,48.01]
meanInt = meanInt_oxy + meanInt_cont
diameter_oxy = [11.724,12.35,12.88,12.31, 10.237]; diameter_cont = [11.49,11.3465,9.8095,9.6875, 9.2509]
diameter_y = diameter_oxy + diameter_cont; diameter = [i * 2 for i in diameter_y]
AspectRatio_oxy = [1.438, 1.564, 1.423, 1.470, 1.733]; AspectRatio_cont = [1.499, 1.252, 1.400, 1.401, 1.377]
AspectRatio = AspectRatio_oxy + AspectRatio_cont

plt.figure(21); plt.clf();
plt.scatter(meanInt_oxy, viability_oxy, color='xkcd:pink')
plt.scatter(meanInt_cont, viability_cont, color='xkcd:green')

corr, _ = pearsonr(viability, AspectRatio);  print(corr)

