import numpy as np
import matplotlib.pyplot as plt
# import PIL.Image
# import imageio
import tifffile
import gc
# import imagej
# import time
# import scyjava
# import pandas
import sys
import os
from tkinter import *
from tkinter import filedialog
from cellpose import denoise
import matplotlib
matplotlib.use("Qt5Agg")


def densityRender(CaptureVol, CellVol, CaptureVolSize, fraction_density):  # count/mL
    counting2d = CaptureVol / CellVol / CaptureVolSize * fraction_density
    return counting2d


# # # parameter initiation
x_num, y_num, z_num = 256, 256, 105
RI = 1.47
CellVol = 708.4  # calculated by manually picking 10 isolated particle images from different data
initialDensity = 24.7 * 1e6  # count/mL, assumed particle density before dilution

# # # # make scatter plot of counting vs density, 4 velocities correspond to 4 colors# # # # # # # #
fraction_density02a = np.array([0.236, 0.2376, 0.2338, 0.2247])  # for label: 0.1, 0.2, 0.4, 0.6
fraction_density010a = np.array([0.1084, 0.1046, 0.0986, 0.0936])
fraction_density0049a = np.array([0.048, 0.0499, 0.0491, 0.0485])
fraction_density0027a = np.array([0.0286, 0.0281, 0.0284, 0.02833])
# flow = [0.1, 0.2, 0.4, 0.6]
density = np.array([0.2, 0.095, 0.049, 0.027]);  actualDensity = initialDensity * density
actualDen = np.tile(actualDensity, (4, 1))
CaptureVol = x_num * y_num * z_num
CaptureVolSize = x_num*4 * y_num*4 * (z_num*5.29 / RI) / 1e12  # um^3 > mL, Z: 5.29 um/pix, X & Y: 4 um/pix

count_0027a = densityRender(CaptureVol, CellVol, CaptureVolSize, fraction_density0027a)
count_0049a = densityRender(CaptureVol, CellVol, CaptureVolSize, fraction_density0049a)
count_010a = densityRender(CaptureVol, CellVol, CaptureVolSize, fraction_density010a)
count_02a = densityRender(CaptureVol, CellVol, CaptureVolSize, fraction_density02a)

fig2 = plt.figure(21);  plt.clf()
ax2 = fig2.subplot_mosaic("A")
ax2['A'].set_xlabel('Density (/mL)')
ax2['A'].set_ylabel('2D counting result (/mL)')

color02 = np.array(['#efc0b2', '#ec9d85', '#f06f47', '#f23a00'])  # density = 0.2a
ax2_a = ax2['A'].scatter(actualDen[:, 0], count_02a, c=color02)
ax2_a = ax2['A'].scatter(actualDen[:, 1], count_010a, c=color02)  # density = 0.1a
ax2_a = ax2['A'].scatter(actualDen[:, 2], count_0049a, c=color02)  # density = 0.05a
ax2_a = ax2['A'].scatter(actualDen[:, 3], count_0027a, c=color02)  # density = 0.02a
ax2_a = ax2['A'].plot(np.linspace(0, actualDensity[0], 10), np.linspace(0, actualDensity[0], 10), 'k--')





