import numpy as np
import matplotlib.pyplot as plt
import tifffile
import gc
import sys
import os
from tkinter import *
from tkinter import filedialog
from cellpose import denoise#, utils, io
import matplotlib
matplotlib.use("Qt5Agg")



# # # # # # # # # - - - - - - - - - 1st bayer demo (Lida) analysis - - - - - - - - - # # # # # # # # #
# # # - - - Day-1 - - - # # #
mix20Via_d1= np.array([69.7, 77, 79.75, 76, 71.8])
mix30Via_d1 = np.array([69.6, 71.4, 79.9, 79.8, 77.8])
mix40Via_d1 = np.array([75.3, 73.1, 85.4, 84.7, 78.3])
mix100Via_d1 = np.array([82.8, 83.7, 82.3, 84.4, 85.1])
# mix20Via_d1= np.array([86.5, 107.3, 125, 102.6]) /255
# mix30Via_d1 = np.array([86.1, 87.9, 108.3, 109.2, 99.6]) /255
# mix40Via_d1 = np.array([123.8, 122.9, 130, 116.1, 116.4]) /255
# mix100Via_d1 = np.array([118.5, 125.35, 109, 123, 125.3]) /255

# # # - - - Day-2 (alive) - - - # # #
mix20Via_d2Liv= np.array([88.7, 88.1, 91.3, 86.6, 83.4])
mix30Via_d2Liv = np.array([88.2, 91.4, 87.1, 89.4, 88.5])
mix40Via_d2Liv = np.array([86.5, 89.5, 89.6, 91.9, 92.1])
mix100Via_d2Liv = np.array([94.2, 95.5, 95.5, 96.2, 94.4])
# mix20Via_d2Liv= np.array([124.7, 119.7, 116.4, 94, 82.1]) /255
# mix30Via_d2Liv = np.array([128, 127.76, 83.3, 109.8, 79.1]) /255
# mix40Via_d2Liv = np.array([100.1, 116.76, 111.4, 125.46, 108.3]) /255
# mix100Via_d2Liv = np.array([139.6, 115, 117.4, 135.1, 146]) /255

# # - - - Day-2 (heated) - - - # # #
mix20Via_d2Dead= np.array([3.5, 4.6, 4.2, 3.9, 5.3])
mix30Via_d2Dead = np.array([2.1, 1.9, 2.7, 2.1, 7.5])
mix40Via_d2Dead = np.array([1, 5, 4.9, 4.4, 3.8])
mix100Via_d2Dead = np.array([3.9, 4.9, 6, 4.1, 4.3])
# mix20Via_d2Dead= np.array([62.7, 84.2, 67.75, 42.5, 77.1]) /255
# mix30Via_d2Dead = np.array([63.88, 58.66, 75.49, 65.6, 82.8]) /255
# mix40Via_d2Dead = np.array([59, 52.4, 40.2, 87.47, 84.5]) /255
# mix100Via_d2Dead = np.array([89.4, 86.26, 68, 86, 90.13]) /255

# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # #
mixProp = np.array([0, 2, 3, 4, 10])  # 20%, 30%, 40%, 100%
viaData_d1 = [mix20Via_d1, mix30Via_d1, mix40Via_d1, mix100Via_d1]
viaData_d2Liv = [mix20Via_d2Liv, mix30Via_d2Liv, mix40Via_d2Liv, mix100Via_d2Liv]
viaData_d2Dead = [mix20Via_d2Dead, mix30Via_d2Dead, mix40Via_d2Dead, mix100Via_d2Dead]

fig1 = plt.figure(10, figsize=(11, 4));  plt.clf()
ax1 = fig1.subplot_mosaic("ab")

# # # - - - - - - - Day-1 vs Day-2 - - - - - - - - # # #
ax1['a'].set_title('Viability over 24h')
ax1['a'].set_ylabel('Measaured viability (%)')
ax1['a'].set_xlabel('Dilute ratio of DA01 (%)')

box1 = ax1['a'].boxplot(viaData_d1, positions=[2, 3, 4, 10], sym='bx', patch_artist=True)#, showmeans=True)
boxColor1 = 'gray'
for median in box1['medians']:  median.set_color(boxColor1)
for whisker in box1['whiskers']:  whisker.set_color(boxColor1)
for cap in box1['caps']:  cap.set_color(boxColor1)
# for mean in box3['means']:
#     mean.set_markeredgecolor(boxColor3);  mean.set_markerfacecolor(boxColor3)
for patch in box1['boxes']:
    patch.set_facecolor(boxColor1);  patch.set_color(boxColor1)
    patch.set_alpha(0.3)
for flier in box1['fliers']:
    flier.set_marker('+')
    flier.set_markeredgecolor(boxColor1)

box2 = ax1['a'].boxplot(viaData_d2Liv, positions=[2, 3, 4, 10], patch_artist=True)#, showmeans=True)
boxColor2 = 'green'
for median in box2['medians']:  median.set_color(boxColor2)
for whisker in box2['whiskers']:  whisker.set_color(boxColor2)
for cap in box2['caps']:  cap.set_color(boxColor2)
# for mean in box3['means']:
#     mean.set_markeredgecolor(boxColor3);  mean.set_markerfacecolor(boxColor3)
for patch in box2['boxes']:
    patch.set_facecolor(boxColor2);  patch.set_color(boxColor2)
    patch.set_alpha(0.3)
for flier in box2['fliers']:
    flier.set_marker('+')
    flier.set_markeredgecolor(boxColor2)

ax1['a'].set_xlim([0, 10.5])
ax1['a'].set_ylim([0, 100])
ax1['a'].set_xticks([0, 2, 3, 4, 10],["0%", "20%", "30%", "40%", "100%"])
ax1['a'].set_yticks([0, 20, 40, 60, 80, 100],["0%", "20%", "40%", "60%", "80%", "100%"])
# ax1['a'].set_xlim([0, 10.5])
# ax1['a'].set_ylim([0, 1])
# ax1['a'].set_xticks([0, 2, 3, 4, 10],["0%", "20%", "30%", "40%", "100%"])
# ax1['a'].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1],["0%", "20%", "40%", "60%", "80%", "100%"])


# # # - - - - - - - Day-2 alive vs dead - - - - - - - - # # #
ax1['b'].set_title('Viability after heat shock')
ax1['b'].set_ylabel('Measaured viability (%)')
ax1['b'].set_xlabel('Dilute ratio of DA01 (%)')

box3 = ax1['b'].boxplot(viaData_d2Dead, positions=[2, 3, 4, 10], patch_artist=True)#, showmeans=True)
boxColor3 = 'red'
for median in box3['medians']:  median.set_color(boxColor3)
for whisker in box3['whiskers']:  whisker.set_color(boxColor3)
for cap in box3['caps']:  cap.set_color(boxColor3)
# for mean in box3['means']:
#     mean.set_markeredgecolor(boxColor3);  mean.set_markerfacecolor(boxColor3)
for patch in box3['boxes']:
    patch.set_facecolor(boxColor3);  patch.set_color(boxColor3)
    patch.set_alpha(0.3)
for flier in box3['fliers']:
    flier.set_marker('+')
    flier.set_markeredgecolor(boxColor3)

box4 = ax1['b'].boxplot(viaData_d2Liv, positions=[2, 3, 4, 10], patch_artist=True)#, showmeans=True)
boxColor4 = 'green'
for median in box4['medians']:  median.set_color(boxColor4)
for whisker in box4['whiskers']:  whisker.set_color(boxColor4)
for cap in box4['caps']:  cap.set_color(boxColor4)
# for mean in box3['means']:
#     mean.set_markeredgecolor(boxColor3);  mean.set_markerfacecolor(boxColor3)
for patch in box4['boxes']:
    patch.set_facecolor(boxColor4);  patch.set_color(boxColor4)
    patch.set_alpha(0.3)
for flier in box4['fliers']:
    flier.set_marker('+')
    flier.set_markeredgecolor(boxColor4)

ax1['b'].set_xlim([0, 10.5])
ax1['b'].set_ylim([0, 100])
ax1['b'].set_xticks([0, 2, 3, 4, 10],["0%", "20%", "30%", "40%", "100%"])
ax1['b'].set_yticks([0, 20, 40, 60, 80, 100],["0%", "20%", "40%", "60%", "80%", "100%"])
# ax1['b'].set_xlim([0, 10.5])
# ax1['b'].set_ylim([0, 1])
# ax1['b'].set_xticks([0, 2, 3, 4, 10],["0%", "20%", "30%", "40%", "100%"])
# ax1['b'].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1],["0%", "20%", "40%", "60%", "80%", "100%"])



# # # # # - - - - [low SNR, trash data] 2st bayer visit (Alan) analysis: mean norm log intensity vs. viability - - - - - # # # # # #
# # # # - - - 10 enface slices were selected in ImageJ, threshold set as auto, measure mean int - - - # # #
# vialB_65via = np.array([55.1, 55.3, 50.2]) /255
# viaA_0via = np.array([47.8, 48.5, 42.3]) /255
# viaC_32via = np.array([48.7, 63.7, 54.3]) /255
# viaD_61via = np.array([41.45, 63, 60.4]) /255
# viaE_51via = np.array([57.8, 48.2, 60.6]) /255
# viaF_37via = np.array([62.7, 49.7, 67.1]) /255
# viaG_68via = np.array([74.8, 64.8]) /255
# viaH_0via = np.array([34.3, 32, 27.7]) /255
# viaI_29via = np.array([65.6, 66.6, 68]) /255
# meanIntList = [viaA_0via, viaH_0via, viaI_29via, viaC_32via, viaF_37via, viaE_51via, viaD_61via, vialB_65via, viaG_68via]
# viaList = np.array([1, 1, 29, 32, 37, 51, 61, 65, 68])
#
# fig1 = plt.figure(10, figsize=(5, 4));  plt.clf()
# ax1 = fig1.subplot_mosaic("a")
# ax1['a'].set_title('Mean normalized log intensity at various viability')
# ax1['a'].set_xlabel('Viability by NC202 (%)')
# ax1['a'].set_ylabel('Mean normalized OCT intensity (a.u.)')
# box1 = ax1['a'].boxplot(meanIntList, positions=[1, 2, 29, 32, 37, 51, 61, 65, 68], widths=2, sym='bx', patch_artist=True)
# ax1['a'].set_xlim([0, 70])
# ax1['a'].set_ylim([0, 0.3])
# ax1['a'].set_xticks([0, 15, 30, 45, 60, 75],["0%", "15%", "30%", "45%", "60%", "75%"])
# ax1['a'].set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],["0", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30"])
# # # # - - - Trash data process, total waste of life - - - # # #



# # # # # # - - - - 2st bayer visit (Alan) analysis: Time lapse over 48h - - - - - # # # #
# vFrac = np.zeros([4, 46])  # vFrac[0, :] is the vFrac' which almost equals to vFrac, vFrac[1, :] is the std of vFrac over 30 depth slices
# vFrac[0, :] = np.array([97, 96.95, 96.84, 97.16, 97.57, 97.59, 97.36, 97.31, 97.11, 97.17, 97.31, 97.45, 97.34,
#                  97.41, 97.53, 97.29, 97.37, 97.36, 97.66, 97.73, 97.81, 97.99,
#                  98.02, 97.86, 98.03, 98.28, 98.51, 98.58, 98.57, 98.48, 98.57, 98.70, 98.82, 98.86, 98.97,
#                  99.03, 99.04, 99.07, 98.97, 99.10, 99.01, 98.99, 99.02, 98.99, 98.88, 98.97])
# vFrac[1, :] = np.array([0.01275689417431349, 0.011610915185692568, 0.012073572714433264, 0.010418950481695376, 0.007813130088803385,
# 0.007374318907053363, 0.008179295295344785, 0.008385419928927771, 0.00812078580514119, 0.00796153136110344, 0.008434330095714202,
# 0.008606546937995448, 0.00856628700975327, 0.008723879506000003,  0.008792933873480117, 0.00841409225340319,  0.008857459156593622,
# 0.00850595419982341,  0.008885614838860162, 0.008816261497571276, 0.008763853811542061, 0.008526613795320553, 0.008156203595076272,
# 0.00799750795761427, 0.008325443163617248, 0.008410770720335888, 0.007794695103337548, 0.00737893843961977, 0.00721948192282,  0.007493953865147953,
# 0.0074991681720682, 0.00724929678244, 0.00703339639001867,  0.00673601081108213, 0.0060612694847513,  0.0059501585208, 0.0059745866,
# 0.00561538143034,  0.00620555804782, 0.005751407698749, 0.00622401404505, 0.00617292971169, 0.005833162467238,  0.0057717575679, 0.0059996293233, 0.0057126423])
# # vFrac[2, :] = np.array([5.6377,5.82898,5.941,6.698,8.2,8.4258,8.4183,8.47179,8.5621,9.47347,9.58912,9.6279,9.609,
# # 9.6066,9.6413,9.4587,9.52345,9.47939,9.60536,9.61763,9.62289,9.635,9.6428,9.646,9.68879,9.78717,9.82179,9.8159,9.8,
# # 9.7828,9.8,9.836,9.8478,9.8542,9.853,9.8726,9.849,9.83175,9.8173,9.8757,9.8256,9.7958,9.8088,9.7596,9.68,9.7
# # ])  # mean of log int in dB
# # vFrac[3, :] = np.array([0.4059,0.417,0.4246,0.3958,0.41848,0.459,0.47169,0.5124,0.5314,0.53857,0.4499,0.40359,0.427,
# # 0.4383,0.3894,0.52786,0.497756, 0.55267,0.4477,0.436,0.4216,0.3877,0.399,0.5,0.4455,0.3419,0.2886,0.277,0.294,
# # 0.35,0.3324,0.286,0.2557,0.2464,0.23,0.2097, 0.21746,0.2209,0.253,0.2036,0.2457,0.28,0.2564,0.2896,0.3573,0.319
# # ])  # std of log int in dB
# timeRec = np.linspace(0, len(vFrac[0, :])-1, len(vFrac[0, :])) + 3
#
# fig3 = plt.figure(15, figsize=(3.5, 3));  plt.clf()
# ax3 = fig3.subplot_mosaic("a")
#
# # ax3['a'].scatter(timeRec, vFrac, color='#282c30', marker='o', s=15)
# ax3['a'].clear()
# ax3['a'].plot(timeRec, vFrac[2, :], color='#203878')
# ax3['a'].fill_between(timeRec, vFrac[0, :] - (vFrac[1, :]*100), vFrac[0, :] + (vFrac[1, :]*100),
#     alpha=0.5, edgecolor='#f5f0f0', facecolor='#bbccfc')
# # ax3['a'].fill_between(timeRec, vFrac[2, :] - (vFrac[3, :]*1), vFrac[2, :] + (vFrac[3, :]*1),
# #     alpha=0.5, edgecolor='#f5f0f0', facecolor='#bbccfc')
#
# ax3['a'].set_title('Time lapse viability over 48h')
# ax3['a'].set_ylabel('Viable fraction (%)')
# ax3['a'].set_xlabel('Time (h)')
# ax3['a'].set_xlim([2.5, 48.5])
# ax3['a'].set_ylim([90, 100])
# # ax3['a'].set_ylim([-5, 25])
# ax3['a'].set_xticks([6, 12, 18, 24, 30, 36, 42, 48])
