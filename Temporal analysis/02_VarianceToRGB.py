from scipy.signal import welch
import gc
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from VLIV.postprocess_vliv import *
from LIV.liv_postprocess import *
from Modules.folder_selection import select_multiple_folders

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

### SYSTEM parameters ###
frames_per_second = 55  # 30
rasterRepeat = 16
rasterRepeat_cal = rasterRepeat  # Can be manually set to a smaller value, e.g. shorter time window.
SYSTEM_NAME = 'ivs800'  # 'ivs800', 'ivs2000'
yasuno_aLiv_swiftness = True
saveImg = True
Is_compute_meanFreq = True  # False: variance-only; True: variance + mean frequency.


### Image processing parameters ###
octRangedB = [-15, 20] if SYSTEM_NAME == 'ivs800' else ([-10, 50] if SYSTEM_NAME == 'ivs2000' else print('Error: set SYSTEM_NAME as ivs800" or "ivs2000"'))
liv_range = (0, 45)
aliv_range = (0, 35);  swift_range = (0, 40)  # (0, 30) todo: swift的range有点怪，可能需要再调调
mliv_range = [0, 0.8]  # LIV (variance): 0~45 / (mLIV: 0~1) / LIV_norm: 0~0.13
meanFreq_range = [0, 5]
Is_display_whenProcess = False
Is_meanFreq_ToSaturationWithLIV = False


""" Choose the parent folder to browse all data folders (Data.bin) """
Fold_list = select_multiple_folders()
extension = ['_IntImg.tif']
DataFold_list = []
for item in Fold_list:
    for n in os.listdir(item):
        if any(x in n for x in extension):
            DataFold_list.append(os.path.join(item, n))
FileNum = len(DataFold_list)


""" Read size of tiff stack """
for FileId in range(FileNum):
    """ read linear intensity data from stack. Dimension (Y, Z, X) """
    DataFold = DataFold_list[FileId]
    DataId_str, extension_str = os.path.basename(DataFold).split('.', 1)
    root = os.path.dirname(DataFold)
    rawDat = tifffile.imread(DataFold)
    # rawDat = tifffile.memmap(DataFold)
    dim_y, dim_z, dim_x = np.shape(rawDat)
    if rasterRepeat > 1:  dim_y_raster = int(dim_y / rasterRepeat)
    elif rasterRepeat == 1:
        dim_y_raster = 1;   rasterRepeat = dim_y
    else:  raise (ValueError('Set rasterRepeat as an integer'))
    # # # - - - initialize variance-to-rgb array, define the display variance range - - - # # #
    batchList = np.linspace(0, dim_y, int(dim_y / rasterRepeat), endpoint=False)
    liv_RgbImg = np.zeros((dim_y_raster, dim_z, dim_x, 3), 'uint8')
    liv_raw = np.zeros((dim_y_raster, dim_z, dim_x), 'float32')
    mliv_raw = np.zeros((dim_y_raster, dim_z, dim_x), 'float32')
    mliv_rgbImg = np.zeros((dim_y_raster, dim_z, dim_x, 3), 'uint8')
    meanFreqImg = np.zeros((dim_y_raster, dim_z, dim_x), 'float32') if Is_compute_meanFreq else None

    if not plt.fignum_exists(1) and Is_display_whenProcess:
        figsize_mag = 3;  fig1 = plt.figure(16, figsize=(figsize_mag, dim_z / dim_x * figsize_mag))
        plt.clf();  ax1 = fig1.subplot_mosaic("a")
    else: pass
    sys.stdout.write('\n')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(0), 0) + ' initialize processing: ' + str(FileId + 1) + '/' + str(FileNum))

    # dim_y_raster = 1
    for batch_id in range(dim_y_raster):
        rawDat_batch = rawDat[(batch_id * rasterRepeat):(batch_id * rasterRepeat + rasterRepeat_cal), :, :]  # linear intensity, dimension: [32(y), 300(z), 256(x)]
        rawDat_batch_log = np.multiply( 10, np.log10(rawDat_batch ) )  # RepeatBscan = 16. Size: [16 (Y), 1024 (Z), 256 (X)]  | DB log intensity Bscan sequence
        batchProj_maxLogInt = np.max(rawDat_batch_log, axis=0)  # average of 16-Bscans. Size: [1024 (Z), 256 (X)]
        batchProj_valMax_clip = np.clip((batchProj_maxLogInt - octRangedB[0]) / (octRangedB[1] - octRangedB[0]), 0, 1)  # Average log intensity > Value
        # plt.figure(13); plt.clf(); plt.imshow(batchProj_valMax, cmap='gray')
        # # # - - - *compute variance/std/freq at each pix > Hue - - - # # #
        batchProj_var = np.var(rawDat_batch_log, axis=0)  # Variance of 16-Bscans. Size: [1024 (Z), 256 (X)]
        batchProj_varHue_clip = np.multiply(np.clip((batchProj_var - liv_range[0]) / (liv_range[1] - liv_range[0]), 0, 1), 0.6)  # Clipped variance > Hue

        """ 自创的positive-scaled LIV，同时转换到hsv > rgb """
        block_logIntensity_positiveScaled = np.multiply(10, np.log10(rawDat_batch + 1))  # non-zero scale factor by log intensity, typical range: (0, 36)
        block_logIntensity_positiveScaled_maxProj = np.max(block_logIntensity_positiveScaled, axis=0)
        block_PositiveLIV = np.var(block_logIntensity_positiveScaled, axis=0)
        batchProj_var_mLIV = block_PositiveLIV / block_logIntensity_positiveScaled_maxProj  # Modified-LIV (dB); typical display range: (0, 1)
        batchProj_var_mLIV_Hueclip = np.multiply(np.clip((batchProj_var_mLIV - mliv_range[0]) / (mliv_range[1] - mliv_range[0]), 0, 1), 0.6)
        mLIV_rgb = hsv_to_rgb(np.transpose([batchProj_var_mLIV_Hueclip, np.ones_like(batchProj_var_mLIV_Hueclip), batchProj_valMax_clip]))
        mliv_rgbImg[batch_id, :, :, :] = np.swapaxes(mLIV_rgb, 0, 1) * 255
        mliv_raw[batch_id, :, :] = batchProj_var_mLIV
        # # # - - - 自创的positive-scaled LIV - - - - # # #

        """ Compute temporal metrics (mean frequency, aliv, swftness) from time sequence, assign to hsv channels """
        # pix_loc = [355, 101]  # [Z_index, X_index]  # 动-green：[385, 52]  静-red：[259, 99]  空-gray: [400, 184]
        if not Is_compute_meanFreq:
            ch_saturation = np.ones_like(batchProj_varHue_clip)
            freq_mean_map = None
            # var_at_pixel = batchProj_var[pix_loc[0], pix_loc[1]]
            # sys.stdout.write("Variance at pixel" + str(pix_loc) + ": " + str(var_at_pixel) + '\n')
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
            if Is_meanFreq_ToSaturationWithLIV:
                batchProj_meanFreqSat_clip = np.clip((freq_mean_map - meanFreq_range[0]) / (meanFreq_range[1] - meanFreq_range[0]), 0, 1)
                ch_saturation = batchProj_meanFreqSat_clip
            else: ch_saturation = np.ones_like(batchProj_varHue_clip)

            # """ Check int fluctuation profile / normalized power spectral density at a designated pixel """
            # # pix_loc = [440, 138]  # [Z_index, X_index]  # 动-green：[385, 52]  静-red：[259, 99]  空-gray: [400, 184]
            # plt_color = 'green'
            # linearIntRecord = rawDat_batch[:, pix_loc[0], pix_loc[1]]
            # # t = np.linspace(0, computeRasterRepeat / frames_per_second, computeRasterRepeat)  # sin wave for plot test
            # # freq_simulate = 5;   linearIntRecord = 1 * np.sin(2 * np.pi * freq_simulate * t) + 1
            # length_fft = round(len(linearIntRecord) / 2)
            # # # welch psd estimation. 分割成多段算psd后平均，牺牲部分频率分辨率换来更好的抗误差，适合多采样点
            # freq_bins, psd = welch(linearIntRecord, fs=frames_per_second, nperseg=seg_size, noverlap=overlap_size, window='hann',
            #                        nfft=len(linearIntRecord) * fft_pad, scaling='density', detrend=False, average='median', return_onesided=True)
            # # # periodogram psd estimation.
            # # freq_bins, psd = periodogram(linearIntRecord, frames_per_second, window='hann',
            # #                              nfft=len(linearIntRecord)*fft_pad, scaling='density', detrend=False, return_onesided=True)
            # # # fft psd estimation.
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

        """convert to hue color space"""
        batchProj_rgb = hsv_to_rgb(np.transpose([batchProj_varHue_clip, ch_saturation, batchProj_valMax_clip]))  # Hue: clipped variance; Saturation: 1;  Value: average log intensity
        liv_RgbImg[batch_id, :, :, :] = np.swapaxes(batchProj_rgb, 0, 1) * 255
        liv_raw[batch_id, :, :] = batchProj_var
        if Is_compute_meanFreq:  meanFreqImg[batch_id, :, :] = freq_mean_map

        """fresh progress bar display"""
        sys.stdout.write('\r')
        j = (batch_id + 1) / dim_y_raster
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j) + ' on batch processing: ' + str(FileId + 1) + '/' + str(FileNum) + ' || DataID: ' + root)
        if Is_display_whenProcess:
            # ax1['a'].clear();  ax1['a'].imshow(np.swapaxes(batchProj_rgb, 0, 1), vmin=0, vmax=1)
            # plt.gca().set_axis_off(); plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            ax1['a'].clear();  ax1['a'].imshow(np.swapaxes(mLIV_rgb, 0, 1), vmin=0, vmax=1)
            plt.pause(0.02)
        else: pass
        # sys.stdout.flush()

    del rawDat
    gc.collect()

    """ Save image and raw data as tiff stacks """
    if saveImg:
        tifffile.imwrite(root + '\\' + DataId_str + '_LIV.tif', liv_RgbImg)
        tifffile.imwrite(root + '\\' + DataId_str + '_LIV_raw.tif', liv_raw)
        tifffile.imwrite(root + '\\' + DataId_str + '_mLIV.tif', mliv_rgbImg)
        tifffile.imwrite(root + '\\' + DataId_str + '_mLIV_raw.tif', mliv_raw)
        try: tifffile.imwrite(root + '\\' + DataId_str + '_meanFreq.tif', meanFreqImg)
        except: pass



    if yasuno_aLiv_swiftness:
        """Compute aliv and swiftness, using https://github.com/ComputationalOpticsGroup/COG-dynamic-OCT-contrast-generation-library. """
        # #todo: Note: >15% pixels with infinity swiftness is usually found.
        frameRepeat = 1;  bscanLocationPerBlock = 1;  frameSeparationTime = 1 / frames_per_second  # frame time interval
        blockRepeat = rasterRepeat  # 16  # Number of block repeats
        blockPerVolume = dim_y_raster  # 256  # Number of blocks in a volume
        vliv_postprocessing(DataFold, "Ibrahim2021BOE", frameSeparationTime, frameRepeat, bscanLocationPerBlock,
                            blockRepeat, blockPerVolume, fitting_method="GPU", motionCorrection=False, save_LivCurve=True,
                            octRange=tuple(octRangedB), alivRange=tuple(aliv_range), swiftRange=swift_range)
    else: pass