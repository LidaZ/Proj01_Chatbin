import os
import sys
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import gc
import matplotlib.cm as cm
# import pingouin as pg
# import pandas as pd
# from scipy.stats import norm
# from scipy.optimize import curve_fit
from lowpass_filter import butter_lowpass_filter
from MapVarToRGB import num_to_rgb
from matplotlib.colors import hsv_to_rgb
import tifffile
import colorsys
from PIL import Image
import matplotlib
matplotlib.use("Qt5Agg")


folderPath = r"F:\Data_2024\20240711_StabilityStudy\20240711_StabilityTest_YeastInRoom2Floor"
stackname = "Storage_20240711_17h45m15s_IntImg"
stackFilePath = folderPath + "\\" + stackname + ".tif"
fs = 50  # Hz, B-scan frequency during acquisition
cutoff = 2;  order = 2  # (cutoff frequency, filtering order), lowpass filter to remove DC component before computing variance
colormap = cm.rainbow
norm = matplotlib.colors.Normalize(vmin=0, vmax=2, clip=True)
rawData = tifffile.imread(stackFilePath)   # load linear intensity data from stack. Dimension (Y, Z, X)
dim_y, dim_z, dim_x = np.shape(rawData)
rawDataRotat = np.swapaxes(rawData, 0, 2)  # rotate en-face plane for 90 degree, so that fast scan (X) is horizontal axis to easy 2D FFT. Dimension (X, Z, Y)
varProj = np.zeros((dim_z, dim_x, 3), 'uint8')


for depthIndex in range(530):   # dim_z
    # depthIndex = 447  # # #
    enfaceData = rawDataRotat[:, depthIndex, :]  # # (dim_x, dim_y)
    enfaceData_lpfilt = butter_lowpass_filter(enfaceData, cutoff, fs, order)
    enfaceData_lp = enfaceData - enfaceData_lpfilt
    # plt.figure(11);    plt.clf();    plt.plot(enfaceData[355, :]);   plt.plot(enfaceData_lp[355, :])

    hueRange = [0, 10]  # variance / std: 0~5
    satRange = [0, 10]  # intensity
    varProj_var = np.var(enfaceData_lp, axis=1)  # np.var()  /  np.std()
    varProj_hue = np.clip((varProj_var - hueRange[0]) / (hueRange[1] - hueRange[0]), 0, 1) * 1
    varProj_proj = np.max(enfaceData, axis=1)  # max projection along Y (time)
    varProj_sat = np.clip((varProj_proj - satRange[0]) / (satRange[1] - satRange[0]), 0, 1) * 1
    varProj_val = np.ones((dim_x), 'uint8') * 1
    varProj_rgb = hsv_to_rgb(np.transpose([varProj_hue, varProj_sat, varProj_val]))
    varProj[depthIndex, :, :] = varProj_rgb * 255

    # varProj_rgb = num_to_rgb(np.var(enfaceData_lp, axis=1), max_val=20)
    # varProj_hsv = colorsys.rgb_to_hsv(varProj_rgb[0], varProj_rgb[1], varProj_rgb[2])
    # varProj[depthIndex, :, ] = np.transpose(varProj_rgb)

    print(str(depthIndex) + '/' + str(dim_z))

img = varProj
figsize_mag = 7
plt.figure(16, figsize=(figsize_mag, dim_z/dim_x*figsize_mag));  plt.clf();  plt.imshow(img, vmin=0, vmax=255)
plt.gca().set_axis_off(); plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
gc.collect()
