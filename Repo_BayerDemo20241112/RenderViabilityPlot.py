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



# # # - - - Day-1 - - - # # #
mix20Via_d1= np.array([69.7, 77, 79.75, 76, 71.8])
mix30Via_d1 = np.array([69.6, 71.4, 79.9, 79.8, 77.8])
mix40Via_d1 = np.array([75.3, 73.1, 85.4, 84.7, 78.3])
mix100Via_d1 = np.array([82.8, 83.7, 82.3, 84.4, 85.1])

# # # - - - Day-2 (alive) - - - # # #
mix20Via_d2Liv= np.array([88.7, 88.1, 91.3, 86.6, 83.4])
mix30Via_d2Liv = np.array([88.2, 91.4, 87.1, 89.4, 88.5])
mix40Via_d2Liv = np.array([86.5, 89.5, 89.6, 91.9, 92.1])
mix100Via_d2Liv = np.array([94.2, 95.5, 95.5, 96.2, 94.4])

# # - - - Day-2 (heated) - - - # # #
mix20Via_d2Dead= np.array([3.5, 4.6, 4.2, 3.9, 5.3])
mix30Via_d2Dead = np.array([2.1, 1.9, 2.7, 2.1, 7.5])
mix40Via_d2Dead = np.array([1, 5, 4.9, 4.4, 3.8])
mix100Via_d2Dead = np.array([3.9, 4.9, 6, 4.1, 4.3])

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
