import sys
import numpy as np
import numpy.fft as fft
# from pyparsing import line_end
from scipy.signal import welch
from scipy.signal import periodogram
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
import matplotlib
from sympy.abc import alpha
from VLIV.postprocess_vliv import *
from LIV.liv_postprocess import *

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

"""
Open source project for cell counting system. 
Generate LIV-encoded images from a 3-D linear intensity volume. 
Author: Lida
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

### System parameters ###
frames_per_second = 55
rasterRepeat = 16
rasterRepeat_cal = rasterRepeat
sys_ivs800 = True
yasuno_aLiv_swiftness = True
saveImg = True
bivar_add_meanFreq = False  # False: variance-only; True: variance + mean frequency.

### Image processing parameters ###
multiFolderProcess = True  # if multiple data folders
live_range_ToHue = [0., 15]  # LIV (variance): 0~12 / mLIV: 0~1 / LIV_norm: 0~0.13
meanFreq_range_ToSat = [0.5, 2.5]
if sys_ivs800:  octRangedB = [-15, 20]  # [-5, 20]
else:  octRangedB = [0, 50]  # set dynamic range of log OCT signal display

### Starts here ###
errorShiftFrame = 0  # 0 for normal, 8 for IVS-800 data
if multiFolderProcess:
    root = tk.Tk();  root.withdraw();  Fold_list = [];  DataFold_list = [];  extension = ['_IntImg.tif']
    folderPath = filedialog.askdirectory(title="Cancel to Stop Enqueue")
    Fold_list.append(folderPath)
    while len(folderPath) > 0:
        folderPath = filedialog.askdirectory(initialdir=os.path.dirname(folderPath), title="Cancel to Stop Enqueue")
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
    rot = tk.Tk();  rot.withdraw();  rot.attributes("-topmost", True);  DataFold_list = filedialog.askopenfilename(filetypes=[("", "*_IntImg.tif")], multiple=True)
    rot.destroy()
    # DataId = os.path.basename(DataFold_list);   root = os.path.dirname(DataFold_list);
    FileNum = np.shape(DataFold_list)[0]
    # if '_IntImg' in DataId:  pass
    # else:  raise(ValueError('Select _IntImg.tif file'))

""" Read size of tiff stack """
for FileId in range(FileNum):
    DataFold = DataFold_list[FileId]
    DataId = os.path.basename(DataFold);  root = os.path.dirname(DataFold)
    rawDat = tifffile.imread(DataFold)  # load linear intensity data from stack. Dimension (Y, Z, X)
    # rawDat = tifffile.memmap(DataFold)
    dim_y, dim_z, dim_x = np.shape(rawDat)
    if rasterRepeat > 1:  dim_y_raster = int(dim_y / rasterRepeat)
    elif rasterRepeat == 1:
        dim_y_raster = 1;   rasterRepeat = dim_y
    else:  raise (ValueError('Set rasterRepeat as an integer'))
    # # # - - - initialize variance-to-rgb array, define the display variance range - - - # # #
    batchList = np.linspace(0, dim_y, int(dim_y / rasterRepeat), endpoint=False)
    varRgbImg = np.zeros((dim_y_raster, dim_z, dim_x, 3), 'uint8')
    varRawImg = np.zeros((dim_y_raster, dim_z, dim_x), 'float32')
    meanFreqImg = np.zeros((dim_y_raster, dim_z, dim_x), 'float32') if bivar_add_meanFreq else None

    if plt.fignum_exists(1):  pass
    else:
        figsize_mag = 3;  fig1 = plt.figure(16, figsize=(figsize_mag, dim_z / dim_x * figsize_mag));  plt.clf()
        ax1 = fig1.subplot_mosaic("a")
    sys.stdout.write('\n')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(0), 0) + ' initialize processing' + ': ' + str(FileId + 1) + '/' + str(FileNum))

    if yasuno_aLiv_swiftness:
        # #todo: compute aliv and swiftness here
        frameRepeat = 1;  bscanLocationPerBlock = 1;  frameSeparationTime = 1 / frames_per_second  # frame time interval
        blockRepeat = rasterRepeat  # 16  # Number of block repeats
        blockPerVolume = dim_y_raster  # 256  # Number of blocks in a volume
        vliv_postprocessing(DataFold, "Ibrahim2021BOE", frameSeparationTime, frameRepeat, bscanLocationPerBlock,
                            blockRepeat, blockPerVolume, fitting_method="GPU", motionCorrection=False,
                            octRange=tuple(octRangedB), alivRange=tuple(live_range_ToHue), swiftRange=(0, 30))
        continue


    # dim_y_raster = 1
    for batch_id in range(dim_y_raster):
        rawDat_batch = rawDat[(batch_id * rasterRepeat + errorShiftFrame):(
                    batch_id * rasterRepeat + errorShiftFrame + rasterRepeat_cal), :, :]  # [32(y), 300(z), 256(x)]
        rawDat_batch_filt = rawDat_batch  # - rawDat_batch_dc        # linear intensity
        rawDat_batch_log = np.multiply( 10, np.log10(rawDat_batch_filt + 1) )  # dB, log intensity, 10*log10(linear), typical range: (0, 36)

        # # # - - - compute max int projection along time window (computeRasterRepeat) at each pix > Value - - - # # #
        batchProj_maxInt = np.max(rawDat_batch_log, axis=0)  # max of log intensity, typical range: (0.1, 36)
        batchProj_valMax_clip = np.clip((batchProj_maxInt - octRangedB[0]) / (octRangedB[1] - octRangedB[0]), 0,1)  # clipped Log int
        # plt.figure(13); plt.clf(); plt.imshow(batchProj_valMax, cmap='gray')

        # # # - - - *compute variance/std/freq at each pix > Hue - - - # # #
        batchProj_var = np.var(rawDat_batch_log, axis=0)  # LIV: linear > log > variance. typical display range: (0, 146) > (0, 10)
        # batchProj_var_norm = batchProj_var  #todo: LIV (dB^2); typical display range: (0, 6)
        batchProj_var_norm = batchProj_var / batchProj_maxInt  #todo: Modified-LIV (dB); typical display range: (0, 1.)
        # batchProj_var_norm = batchProj_var / (batchProj_maxInt ** 2)  #todo: Normalized LIV (a.u.); typical display range: (0, 1.3)
        batchProj_varHue_clip = np.multiply(np.clip((batchProj_var_norm - live_range_ToHue[0]) / (live_range_ToHue[1] - live_range_ToHue[0]), 0, 1), 0.6)  # limit color display range from red to blue


        """ Compute temporal metrics (mean frequency, aliv, swftness) from time sequence, assign to hsv channels """
        pix_loc = [355, 101]  # [Z_index, X_index]  # 动-green：[385, 52]  静-red：[259, 99]  空-gray: [400, 184]
        if not bivar_add_meanFreq:
            ch_saturation = np.ones_like(batchProj_varHue_clip)
            freq_mean_map = None
            var_at_pixel = batchProj_var_norm[pix_loc[0], pix_loc[1]]
            sys.stdout.write("Variance at pixel" + str(pix_loc) + ": " + str(var_at_pixel) + '\n')
        else:
            # # # - - -  [2026/02/24: obsolete due to negligible contrast improvement] Compute mean frequency of each B-scan (from normalized power spectral density) > saturation - - - # # #
            fft_pad = 2;  seg_size = int(rasterRepeat_cal / 1);  overlap_size = int(seg_size / 2)
            # # compute normalized power spectral density and mean frequency for all pixels in the Z-X plane
            t_len, z_len, x_len = rawDat_batch.shape  # rawDat_batch: shape [time, Z, X]
            linearIntRecord_all = rawDat_batch.reshape(t_len, -1)  # reshape to [time, N_pixel]
            freq_bins, psd_all = welch(linearIntRecord_all, fs=frames_per_second, nperseg=seg_size, noverlap=overlap_size, window="hann",
                                       nfft=t_len * fft_pad, scaling="density", detrend=False, average="median", axis=0)  # psd_all: shape [n_freq, N_pixel]
            psd_sum = np.sum(psd_all, axis=0, keepdims=True)  # shape [n_freq, N_pixel] -> [1, N_pixel]
            psd_sum[psd_sum == 0] = 1.0  # set power over spectrum as 1 when computed as 0, to avoid dividing by 0 in norm
            psd_norm_all = psd_all / psd_sum  # L1 normalize PSD at each pixel. Ref: doi.org/10.1038/s41377-020-00375-8
            freq_mean_all = freq_bins @ psd_norm_all  # freq_bins: [n_freq], psd_norm_all: [n_freq, N_pixel]
            # # autocorrelation to find dominant frequency: https://stackoverflow.com/questions/78089462/how-to-extract-dominant-frequency-from-numpy-array
            freq_mean_map = freq_mean_all.reshape(z_len, x_len)  # reshape 回 (Z, X)，得到整幅图的 freq_mean 分布
            batchProj_meanFreqSat_clip = np.clip( (freq_mean_map - meanFreq_range_ToSat[0]) / (meanFreq_range_ToSat[1] - meanFreq_range_ToSat[0]), 0, 1)
            ch_saturation = batchProj_meanFreqSat_clip

            # """ Check int fluctuation profile / normalized power spectral density at a designated pixel """
            # # pix_loc = [440, 138]  # [Z_index, X_index]  # 动-green：[385, 52]  静-red：[259, 99]  空-gray: [400, 184]
            # plt_color = 'green'
            # linearIntRecord = rawDat_batch[:, pix_loc[0], pix_loc[1]]
            # # t = np.linspace(0, computeRasterRepeat / frames_per_second, computeRasterRepeat)  # sin wave for plot test
            # # freq_simulate = 5;   linearIntRecord = 1 * np.sin(2 * np.pi * freq_simulate * t) + 1
            # length_fft = round(len(linearIntRecord) / 2)
            # # #todo: welch psd estimation. 分割成多段算psd后平均，牺牲部分频率分辨率换来更好的抗误差，适合多采样点
            # freq_bins, psd = welch(linearIntRecord, fs=frames_per_second, nperseg=seg_size, noverlap=overlap_size, window='hann',
            #                        nfft=len(linearIntRecord) * fft_pad, scaling='density', detrend=False, average='median', return_onesided=True)
            # # #todo: periodogram psd estimation.
            # # freq_bins, psd = periodogram(linearIntRecord, frames_per_second, window='hann',
            # #                              nfft=len(linearIntRecord)*fft_pad, scaling='density', detrend=False, return_onesided=True)
            # # #todo: fft psd estimation.
            # # psd_fft = fft.fft(linearIntRecord, n=len(linearIntRecord) * fft_pad) ** 2 / (seg_size * frames_per_second)
            # # n_half = int(seg_size * fft_pad / 2 + 1)
            # # psd = np.abs(psd_fft[:n_half]) ** 2
            # # freq_bins = np.fft.rfftfreq(seg_size*fft_pad, 1 / frames_per_second)
            #
            # psd_norm = psd / np.sum(psd)  # L1 normalized psd, its sum over all frequencies (freq_bins) is 1.
            # freq_mean = psd_norm.dot(freq_bins)
            # if not plt.fignum_exists(14):
            #     fig2 = plt.figure(14, figsize=(5, 6));  plt.clf()
            #     ax2 = fig2.subplot_mosaic("a;b");  fig2.tight_layout()
            #     ax2['a'].cla()
            #     ax2['a'].set_xlim(xmin=0)
            #     ax2['a'].set_xticks([0, length_fft, length_fft * 2], ["0", f"{(length_fft / frames_per_second):.1f}", f"{(2 * length_fft / frames_per_second):.1f}"])
            #     ax2['a'].set_xlabel('Time (s)', labelpad=0)
            #     ax2['a'].set_ylabel('Linear intensity w/o BG (a.u.)')
            #     ax2['b'].cla()
            #     ax2['b'].set_xlim(xmin=0, xmax=freq_bins[-1])
            #     ax2['b'].set_ylim(0, 1)  # psd_norm.max())  # 直接画PSD，总能量和光强int线性相关
            #     ax2['b'].set_xlabel('Frequency (Hz)', labelpad=0)
            #     ax2['b'].set_ylabel('L1 Norm PSD (a.u.)')
            #     ax2['b'].set_yscale('log');  ax2['b'].set_ylim(0.001, 10)
            # ax2['a'].plot(linearIntRecord, color=plt_color, alpha=0.7)  # ax2['a'].plot(np.log10(linearIntRecord))
            # ax2['b'].fill_between(freq_bins, psd_norm, color=plt_color, alpha=0.3);  # ax2['b'].set_ylim([0, 0.5])  # 画L1 normalized PSD
            # ax2['b'].axvline(x=freq_mean, color=plt_color, linestyle='--', label=f'Mean freq: {freq_mean:.1f} Hz')
            # try: lgd.remove();
            # except NameError: pass
            # lgd = ax2['b'].legend()

        # #todo: convert to hue color space
        batchProj_rgb = hsv_to_rgb(np.transpose([batchProj_varHue_clip,
                                                 ch_saturation,
                                                 batchProj_valMax_clip]))  # [varProj_hue, varProj_sat/_val, varProj_val]  # [batchProj_varHue, np.ones_like(batchProj_varHue), batchProj_valMax_clip]
        varRgbImg[batch_id, :, :, :] = np.swapaxes(batchProj_rgb, 0, 1) * 255
        varRawImg[batch_id, :, :] = batchProj_var_norm
        if bivar_add_meanFreq:  meanFreqImg[batch_id, :, :] = freq_mean_map

        # #todo: fresh progress bar display
        sys.stdout.write('\r')
        j = (batch_id + 1) / dim_y_raster
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j) + ' on batch processing' + ': '
                         + str(FileId + 1) + '/' + str(FileNum) + ' || DataID: ' + root)
        ax1['a'].clear();  ax1['a'].imshow(np.swapaxes(batchProj_rgb, 0, 1), vmin=0, vmax=1)
        # plt.gca().set_axis_off(); plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.pause(0.02)

    del DataFold, rawDat
    gc.collect()

    """ Save image and raw data as tiff stacks """
    if saveImg:
        tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + 'LIV.tif', varRgbImg)
        tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + 'LIV_raw.tif', varRawImg)
        try:
            tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + 'meanFreq.tif', meanFreqImg)
        except:
            pass
