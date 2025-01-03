# import os
import sys
import numpy as np
# from numpy import fft
import gc
# import matplotlib.cm as cm
# from tqdm.auto import tqdm
# import pandas as pd
# from scipy.stats import norm
# from scipy.optimize import curve_fit
# from lowpass_filter import butter_lowpass_filter
# from MapVarToRGB import num_to_rgb
from matplotlib.colors import hsv_to_rgb
import tifffile
import os
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

"""
Open source project for cell counting system. 
Generate LIV-encoded images from a 3-D linear intensity volume. 
Author: Yijie, Lida
Convertion is performed in HSV color space. 
Hue: Normalized log intensity variance. 
    where normalization is performed as log-intensity-variance divided by 
    max-log-intensity (during a 32-frame sequence)
Saturation: set as one
Brightness: max-log-intensity

How to use:
Run > select a _IntImg.tif file 

Para setting: 
if using IVS-800 data: 
    sys_ivs800 = True
"""


###

# # # - - - [1],[2, 33多一帧], [34, 65], [66, 97], [98, 129]..., [3938, 3969], [3970, 4000少一帧]- - - # # #
errorShiftFrame = 0  # = 1 before 2024/09/05. Bug in scan pattern was fixed.
sys_ivs800 = True
rasterRepeat = 32
computeRasterRepeat = 32
saveImg = True


hueRange = [0., 1]  # LIV (variance): 0~30 / LIV_norm: 0~1
octRangedB = [0, 50]  # set dynamic range of log OCT signal display
if sys_ivs800:  octRangedB = [-5, 25]


tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True); stackFilePath = filedialog.askopenfilename(filetypes=[("", "*_IntImg.tif")])
DataId = os.path.basename(stackFilePath);   root = os.path.dirname(stackFilePath);  tk.destroy()
if '_IntImg' in DataId:  pass
else:  raise(ValueError('Select _IntImg.tif file'))
print('Loading data folder: ' + root)
fs = 50  # Hz, B-scan frequency during acquisition
cutoff = 0.5;  order = 1  # (cutoff frequency = 0.5, filtering order = 2), lowpass filter to remove DC component before computing variance
# colormap = cm.rainbow
# norm = matplotlib.colors.Normalize(vmin=0, vmax=2, clip=True)

# # # - - - read size of tiff stack - - - # # #
rawDat = tifffile.imread(stackFilePath)   # load linear intensity data from stack. Dimension (Y, Z, X)
# rawDat = tifffile.memmap(stackFilePath)
dim_y, dim_z, dim_x = np.shape(rawDat)
if rasterRepeat > 1:
    dim_y_raster = int(dim_y / rasterRepeat)
elif rasterRepeat == 1:
    dim_y_raster = 1
    rasterRepeat = dim_y
# # # - - - initialize variance-to-rgb array, define the display variance range - - - # # #
batchList = np.linspace(0, dim_y, int(dim_y/rasterRepeat), endpoint=False)
varRgbImg = np.zeros((dim_y_raster, dim_z, dim_x, 3), 'uint8')
batchProj_sat = np.ones((dim_z, dim_x), 'float32')
varRawImg = np.zeros((dim_y_raster, dim_z, dim_x), 'float32')


# dim_y_raster = 1
for batch_id in range(dim_y_raster):
    # # # - - - filt dc component, extract fluctuation with f>0.5hz when fs=50hz - - - # # #
    rawDat_batch = rawDat[(batch_id*rasterRepeat+errorShiftFrame):(batch_id*rasterRepeat+errorShiftFrame+computeRasterRepeat), :, :]  # [32(y), 300(z), 256(x)]
    # rawDat_batch_dc = butter_lowpass_filter(rawDat_batch, cutoff, fs, order, 0)
    # # # - - - disable DC component filter function for now - - - # # #
    rawDat_batch_filt = rawDat_batch  # - rawDat_batch_dc        # linear intensity

    rawDat_batch_log = np.multiply(10, np.log10(rawDat_batch_filt + 1))  # dB, log intensity, 10*log10(linear), typical range: (0, 36)

    # # # - - - compute max int projection at each pix > Value - - - # # #
    batchProj_valMax = np.max(rawDat_batch_log, axis=0)  # max of log intensity, typical range: (0.1, 36)
    batchProj_valMax_clip = np.clip((batchProj_valMax-octRangedB[0]) / (octRangedB[1]-octRangedB[0]), 0, 1)   # clipped Log int
    # plt.figure(13); plt.clf(); plt.imshow(batchProj_val, cmap='gray')

    # # # - - - compute variance/std/freq at each pix > Hue - - - # # #
    batchProj_var = np.var(rawDat_batch_log, axis=0)  # LIV: linear > log > variance. typical range: (0, 146) > (0, 30)
    batchProj_var_norm = batchProj_var / batchProj_valMax  # typical range: (0, 1.3).
    # batchProj_var = np.var(rawDat_batch_log / (np.tile(batchProj_valMax, (32, 1, 1))), axis=0)
    # batchProj_var_norm = batchProj_var
    # # #
    batchProj_varHue = np.multiply(np.clip(
        (batchProj_var_norm-hueRange[0]) / (hueRange[1]-hueRange[0]), 0, 1), 0.6)  # limit color display range from red to blue

    # # # - - - convert to hue color space - - - # # #
    batchProj_rgb = hsv_to_rgb(
        np.transpose([batchProj_varHue, batchProj_sat, batchProj_valMax_clip]))  # [varProj_hue, varProj_sat/_val, varProj_val]
    varRgbImg[batch_id, :, :, :] = np.swapaxes(batchProj_rgb, 0, 1) * 255

    varRawImg[batch_id, :, :] = batchProj_var_norm
    # # # - - - fresh progress bar display - - - # # #
    sys.stdout.write('\r')
    j = (batch_id + 1) / dim_y_raster
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j) + ' on batch processing')

    figsize_mag = 3
    plt.figure(16, figsize=(figsize_mag, dim_z/dim_x*figsize_mag));  plt.clf()
    plt.imshow(np.swapaxes(batchProj_rgb, 0, 1), vmin=0, vmax=1)
    plt.gca().set_axis_off(); plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.pause(0.01)

    # # # - - - check int fluctuation profile at designated pixel - - - # # #
    # pix_loc = [250, 60]  # [Y_index, X_index]
    # plt.figure(14); plt.clf(); plt.plot(rawDat_batch[:, pix_loc[0], pix_loc[1]])
    # plt.plot(rawDat_batch_dc[:, pix_loc[0], pix_loc[1]])
    # plt.plot(rawDat_batch_filt[:, pix_loc[0], pix_loc[1]])
    # print('var is: ', str(np.var(rawDat_batch_filt[:, pix_loc[0], pix_loc[1]])))

gc.collect()

# # - - - save image as tiff stack - - - # # #
if saveImg:
    tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + 'LIV.tif', varRgbImg)
    tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + 'LIV_raw.tif', varRawImg)
