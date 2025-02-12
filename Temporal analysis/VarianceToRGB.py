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
import tkinter as tk
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
rasterRepeat = 32
computeRasterRepeat = 32
sys_ivs800 = True
saveImg = True

multiFolderProcess = False  # if multiple data folders
hueRange = [0., 1]  # LIV (variance): 0~30 / LIV_norm: 0~1
if sys_ivs800: octRangedB = [-5, 25]
else: octRangedB = [0, 50]  # set dynamic range of log OCT signal display

if multiFolderProcess:
    root = tk.Tk(); root.withdraw(); Fold_list = []; DataFold_list = []; extension = ['_IntImg.tif']
    folderPath = filedialog.askdirectory()
    Fold_list.append(folderPath)
    while len(folderPath) > 0:
        folderPath = filedialog.askdirectory(initialdir=os.path.dirname(folderPath))
        if not folderPath: break
        Fold_list.append(folderPath)
    for item in Fold_list:  # list all files contained in each folder
        fileNameList = os.listdir(item)
        for n in fileNameList:
            if any(x in n for x in extension):
                DataFold_list.append(os.path.join(item, n))
    FileNum = len(DataFold_list)
    root.destroy()
else:
    rot = tk.Tk(); rot.withdraw(); rot.attributes("-topmost", True);
    DataFold_list = filedialog.askopenfilename(filetypes=[("", "*_IntImg.tif")], multiple = True)
    rot.destroy()
    # DataId = os.path.basename(DataFold_list);   root = os.path.dirname(DataFold_list);
    FileNum = np.shape(DataFold_list)[0]
    # if '_IntImg' in DataId:  pass
    # else:  raise(ValueError('Select _IntImg.tif file'))
# fs = 50  # Hz, B-scan frequency during acquisition
# cutoff = 0.5;  order = 1  # (cutoff frequency = 0.5, filtering order = 2), lowpass filter to remove DC component before computing variance
# colormap = cm.rainbow
# norm = matplotlib.colors.Normalize(vmin=0, vmax=2, clip=True)

# # # - - - read size of tiff stack - - - # # #
for FileId in range(FileNum):
    DataFold = DataFold_list[FileId]
    DataId = os.path.basename(DataFold);   root = os.path.dirname(DataFold)
    print('Loading data folder: ' + root)
    rawDat = tifffile.imread(DataFold)   # load linear intensity data from stack. Dimension (Y, Z, X)
    # rawDat = tifffile.memmap(DataFold)
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

    if 'fig1' in globals(): pass
    else:
        figsize_mag = 3
        fig1 = plt.figure(16, figsize=(figsize_mag, dim_z / dim_x * figsize_mag));   plt.clf()
        ax1 = fig1.subplot_mosaic("a")
    sys.stdout.write('\n')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(0), 0) + ' initialize processing' + ': ' + str(FileId + 1) + '/' + str(FileNum))
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
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j) + ' on batch processing' + ': ' + str(FileId + 1) + '/' + str(FileNum))
        ax1['a'].clear();  ax1['a'].imshow(np.swapaxes(batchProj_rgb, 0, 1), vmin=0, vmax=1)
        # plt.gca().set_axis_off(); plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.pause(0.02)

        # # # - - - check int fluctuation profile / frequency spec at a designated pixel - - - # # #
        # pix_loc = [401, 131]  # [Y_index, X_index]
        # lineProfile = rawDat_batch[:, pix_loc[0], pix_loc[1]]
        # lineProfile_freq = np.abs(fft.fft(lineProfile))
        # lineProfile_freqNorm = lineProfile_freq / np.max(lineProfile_freq)
        # length_fft = round(len(lineProfile)/2)
        #
        # fig1 = plt.figure(14, figsize=(7, 7));  plt.clf()
        # ax1 = fig1.subplot_mosaic("a;b")
        # ax1['a'].cla()
        # ax1['a'].plot(lineProfile)
        # ax1['a'].set_xticks([0, 10, 20, 30], ["0", "0.2", "0.4", "0.6"])
        # ax1['a'].set_xlabel('Time (s)')
        # ax1['b'].cla()
        # ax1['b'].plot(lineProfile_freqNorm[0:length_fft])
        # ax1['b'].set_xticks([0, 4, 8, 12, 16], ["0", "6.25", "12.5", "18.75", "25.0"])  # 50Hz/32
        # ax1['b'].set_xlabel('Frequency (Hz)')
    del DataFold, rawDat
    gc.collect()

    # # - - - save image as tiff stack - - - # # #
    if saveImg:
        tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + 'LIV.tif', varRgbImg)
        tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + 'LIV_raw.tif', varRawImg)
