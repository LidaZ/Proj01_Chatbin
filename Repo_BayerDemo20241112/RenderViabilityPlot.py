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
mix30Via_d2Liv = np.array([88.2, 91.4, 86, 89.4, 88.5])
mix40Via_d2Liv = np.array([86.5, 89.5, 89.6, 91.9, 92.1])
mix100Via_d2Liv = np.array([94.2, 95.5, 95.5, 96.2, 94.4])

# # # - - - Day-2 (heated) - - - # # #
# mix20Via_d2Dead= np.array([11.2, ])
# mix30Via_d2Dead = np.array([ ])
# mix40Via_d2Dead = np.array([ ])
# mix100Via_d2Dead = np.array([3.9, 4.9, 6, 4.1, 4.3])

# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # #
mixProp = np.array([0, 2, 3, 4, 10])  # 20%, 30%, 40%, 100%
viaData_d1 = [mix20Via_d1, mix30Via_d1, mix40Via_d1, mix100Via_d1]
viaData_d2Liv = [mix20Via_d2Liv, mix30Via_d2Liv, mix40Via_d2Liv, mix100Via_d2Liv]
# if86Da02 = [86, 87.8, 88.7, 89.6, 95]
# ifDeadDa02 = [0, 19, 28.5, 38, 95]

fig1 = plt.figure(10, figsize=(6, 4));  plt.clf()
ax1 = fig1.subplot_mosaic("a")
ax1['a'].set_ylabel('Measaured viability (%)')
ax1['a'].set_xlabel('Dilute ratio of DA01 (%)')

# # # - - - - - - - Day-1 vs Day-2 - - - - - - - - # # #
box1 = ax1['a'].boxplot(viaData_d1, positions=[2, 3, 4, 10], sym='bx', patch_artist=True, showmeans=True)
for median in box1['medians']:  median.set_color('black')
for patch in box1['boxes']:
    patch.set_facecolor('blue')
    patch.set_alpha(0.3)

box2 = ax1['a'].boxplot(viaData_d2Liv, positions=[2, 3, 4, 10], sym='rx', patch_artist=True, showmeans=True)
for median in box2['medians']:  median.set_color('black')
for patch in box2['boxes']:
    patch.set_facecolor('green')
    patch.set_alpha(0.3)
# ax1['a'].plot(mixProp, if86Da02, 'b--')
# ax1['a'].plot(mixProp, ifDeadDa02, 'k:')

# ax1['a'].set_ylim([40, 100])
ax1['a'].set_xlim([0, 10.5])
plt.xticks([0, 2, 3, 4, 10],["0%", "20%", "30%", "40%", "100%"])
plt.yticks([0, 20, 40, 60, 80, 100],["0%", "20%", "40%", "60%", "80%", "100%"])