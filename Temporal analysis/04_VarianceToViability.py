# import os
import sys
# import time
import numpy as np
# from numpy import fft
# import gc
# import matplotlib.cm as cm
# from tqdm.auto import tqdm
# import pandas as pd
# from scipy.stats import norm
# from scipy.optimize import curve_fit
# from lowpass_filter import butter_lowpass_filter
# from MapVarToRGB import num_to_rgb
# from matplotlib.colors import hsv_to_rgb
# import imagej
import tifffile
import os
from tkinter import *
from tkinter import filedialog
import matplotlib
from torch.utils.tensorboard.summary import histogram

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
# from mpl_point_clicker import clicker
from matplotlib.backend_bases import MouseButton
import matplotlib.patches as patches
from cellpose import denoise #, utils, io
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap



"""
# Open source project for cell counting system. 
# Compute viability based on the normalized LIV by calculating the volume fraction above and under a given threshold. 
# Author: Lida
# 
# Parameters: 
# zSlice: depth range of Z-stack images for computing viability
# intThreshold: normalized log intensity threshold to mask out cell regions. Note that the log intensity image (en-face) is the maximum log intensity projection along each raster period, then being denoised by Cellpose v3 using 'cyto3' model.
# viabilityThreshold: LIV threshold to determine whether the LIV value in a pixel represents Living / dead. 
# 
# How to use:
# 1. run 01_ImageConverter.py, to convert raw data (log int) to linear and log int image stacks (Data_IntImg.tif and Data_3d_view.tif). 
# 2. run 02_VarianceToRGB.py, to encode temporal variance of log int as Hue, max log int (during raster period) as Value, 1 as Saturation. 
# 3. open Data_IntImg_LIV.tif in ImageJ, and measure the tilting angle along X (Bscan_tilt) and Y (Y_tilt)
# 4. run /Fiji_macro/AutoRotateMacro.ijm, manually set 'Bscan_tilt' and 'Y_tilt' from the above measurements, and process all 3 image stacks. 
# 5. open aligned LIV image (Data_IntImg_LIV.tif), and select the depth range for computing viability. 
# 6. run 04_VarianceToViability.py, manually set 'zSlice' to be the determined depth range, and select the LIV image (Data_IntImg_LIV.tif) to start computing viability. 
# 
# Para setting: 
"""


def on_press(event):
    global start_point, rect
    if event.button is MouseButton.LEFT:
        try:
            start_point = (int(event.xdata), int(event.ydata))
        except TypeError:
            raise ValueError('Select within the frame')
        # print(f'start {int(event.xdata)} {int(event.ydata)}')
        rect = patches.Rectangle((start_point[0], start_point[1]), 0, 0,
                                 linewidth=1, edgecolor='y', facecolor='none')
        event.inaxes.add_patch(rect)

def on_drag(event):
    if start_point is not None and event.button is MouseButton.LEFT:
        try:
            end_point = (int(event.xdata), int(event.ydata))
        except TypeError:
            end_point = start_point
        width = end_point[0] - start_point[0]
        height = end_point[1] - start_point[1]
        rect.set_width(width);  rect.set_height(height);   rect.set_xy((start_point[0], start_point[1]))
        # print('dragging')

def on_release(event):
    global start_point, end_point, rectangle_coords, selectNotDone
    if start_point is not None and event.inaxes is not None:
        end_point = (int(event.xdata), int(event.ydata))
        # print(f'release {int(event.xdata)} {int(event.y)}')
        x1, y1 = start_point;    x2, y2 = end_point
        rectangle_coords = [
            (x1, y1),  # top-left
            (x1, y2),  # bottom-left
            (x2, y1),  # top-right
            (x2, y2)  # bottom-right
        ]
        # print("Rectangle corners (in pixels):", rectangle_coords)
        event.inaxes.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='y', facecolor='none'))
        selectNotDone = False
        return rectangle_coords

def findOverlapRect(firstFrameCord, lastFrameCord):
    # firstFrameCord, lastFrameCord = (x, y)
    LeftUp = (max(firstFrameCord[0][0], lastFrameCord[0][0]), max(firstFrameCord[0][1], lastFrameCord[0][1]))
    LeftDown = (LeftUp[0], min(firstFrameCord[1][1], lastFrameCord[1][1]))
    RightUp = (min(firstFrameCord[2][0], lastFrameCord[2][0]), max(firstFrameCord[2][1], lastFrameCord[2][1]))
    RightDown = (RightUp[0], min(firstFrameCord[3][1], lastFrameCord[3][1]))
    return [LeftUp, LeftDown, RightUp, RightDown]

def drawRectFromFrame(ax1, fig1, rawDat, frameId):
    global start_point, end_point, rect, rectangle_coords, selectNotDone
    _ = ax1.imshow(rawDat[frameId, ...])  # rawDat[frameId, dim_z, dim_x, ch]
    start_point = None;  end_point = None;  rect = None;  selectNotDone = True;  rectangle_coords = []
    cidPress = fig1.canvas.mpl_connect('button_press_event', on_press)  # connect the event signal (press, drag, release) to the callback functions
    cidDrag = fig1.canvas.mpl_connect('motion_notify_event', on_drag)
    cidRelease = fig1.canvas.mpl_connect('button_release_event', on_release)
    while selectNotDone:  plt.pause(0.3)
    fig1.canvas.mpl_disconnect(cidPress)  # disconnect event handler, using the same cid
    fig1.canvas.mpl_disconnect(cidDrag)
    fig1.canvas.mpl_disconnect(cidRelease)
    FrameCoord = rectangle_coords; print('selected frame: ', str(frameId), ' ROI is: ', FrameCoord)
    return FrameCoord


x_meanFreq_tmp = []
y_Liv_tmp = []


zSlice = [327, 328]  # manual z slicing range to select depth region for computing viability
intThreshold = 0.35
viabilityThreshold = 0.18
bivariate_mode = True  # Enalbe bivariate analysis using mean frenquency and (modified) LIV
manual_pick = False  # Enable manual pixel labeling on ax1['a']. Automatic display all masked pixels when set to False.
# viaIntThreshold = 13  # bullshit threshold on intensity to compute viability
VolFlip = False

tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True); stackFilePath = filedialog.askopenfilename(filetypes=[("", "*_LIV.tif")])
DataId = os.path.basename(stackFilePath);   root = os.path.dirname(stackFilePath);  tk.destroy()
print('Loading fileID: ' + stackFilePath)

#todo:  read size of tiff stack
if VolFlip:  rawDat = tifffile.imread(stackFilePath)
else:
    memmap_rawDat = tifffile.memmap(stackFilePath, mode='r')
    rawDat = np.swapaxes(memmap_rawDat, 0, 1)   #todo: load linear intensity data from stack. Dimension (Y, Z, X)
dim_z, dim_y, dim_x = np.shape(rawDat)[0:3]  #todo: [dim_y, dim_z, dim_x] original dimensions before applying `AutoRotateMacro.ijm`
cropCube = np.zeros([dim_z, dim_y, dim_x], dtype=int)
model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")
zSliceList = np.linspace(zSlice[0], zSlice[1], zSlice[1]-zSlice[0]+1).astype('int')

if plt.fignum_exists(1):  pass  # plt.clf()  # pass
else:
    fig1 = plt.figure(1, figsize=(9, 4.5))
    ax1 = fig1.subplot_mosaic("aaabc;aaadd;aaadd", per_subplot_kw={"d": dict(projection='scatter_density')})
    ax1['a'].title.set_text('Drag rectangle to select ROI from dOCT')
    ax1['b'].title.set_text('After manual cropping')
    ax1['c'].title.set_text('Segmentation mask')
    if bivariate_mode is False:
        ax1['d'].set_ylabel('Viable fraction');  ax1['d'].set_xlabel('En-face slice at depth (um)')
        ax1['d'].set_ylim([-0.01, 1.02]);  ax1['d'].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax1['d'].set_xlim([0, len(zSliceList)*2]);
    else:
        ax1['d'].set_xlabel('Mean frequency (Hz)');  ax1['d'].set_ylabel('LIV (dB$^2$)')
        # ax1['d'].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 1, 2, 3, 4, 5])  # rescale from mLIV (dB) to LIV (dB^2)

#todo: draw rectangles at the first and last frames, and the overlapping cubic is the 3D ROI for viability (volume fraction) computation
frameIndex = zSliceList[0]
firstFrameCord = drawRectFromFrame(ax1['a'], fig1, rawDat, frameIndex)
frameIndex = zSliceList[-1]
lastFrameCord = drawRectFromFrame(ax1['a'], fig1, rawDat, frameIndex)

#todo: create 3D mask from the two rectangles' coordinates
overlapRect = findOverlapRect(firstFrameCord, lastFrameCord)
# overlapRect = findOverlapRect([(2, 1), (2, 237), (254, 1), (254, 237)], [(3, 3), (3, 197), (251, 3), (251, 197)])
cropCube[:, overlapRect[0][1]:overlapRect[1][1], overlapRect[0][0]:overlapRect[2][0]] = 1  # dim_z, dim_y, dim_x

#todo: load linear intensity stack, apply `cropCube`, denoising using `Cellpose v3`, apply `intThreshold` to segment cell regions.
# 3D version is only for test visualize, a 2D version is preferred which is supposed to work with an en-face image stack
linIntFilePath = root + '/' + DataId[:-15] + '_3d_view.tif'
rawLivFilePath = root + '/' + DataId[:-15] + '_IntImg_LIV_raw.tif'
if bivariate_mode:  meanFreqFilePath = root + '/' + DataId[:-15] + '_IntImg_meanFreq.tif'


viabilityList = [];  viaList = [];  totalList = []
tmpList = []
pts_lime = [];  pts_magenta = []
if VolFlip is not True:
    memmap_logInt = tifffile.memmap(linIntFilePath, mode='r')
    memmap_rawLiv = tifffile.memmap(rawLivFilePath, mode='r')
    if bivariate_mode:  memmap_meanFreq = tifffile.memmap(meanFreqFilePath, mode='r')

for frameIndex in zSliceList:
    if VolFlip:  logIntFrame = tifffile.imread(linIntFilePath, key=frameIndex)
    else:  logIntFrame = memmap_logInt[:, frameIndex, :]
    # ax1['a'].clear();  ax1['a'].imshow(logIntFrame, cmap='gray')
    rawDat_enfaceSlice = rawDat[frameIndex, ...]; 
    ax1['a'].clear();  ax1['a'].imshow(rawDat_enfaceSlice)  # cropIntFrame = logIntFrame.copy() * cropCube[frameIndex, ...]
    cropIntFrame = logIntFrame * cropCube[frameIndex, ...]  # margin zeros should not be passed to cellpose, otherwise indexing error will raise
    cropIntFrame_seg = cropIntFrame[overlapRect[0][1]:overlapRect[1][1], overlapRect[0][0]:overlapRect[2][0]]
    ax1['b'].clear();  ax1['b'].imshow(cropIntFrame, cmap='gray')
    try:
        _, _, _, cropIntFrame_seg_dn = model.eval(cropIntFrame_seg, diameter=None, channels=[0, 0]) # , niter=200000)
        _ = cropCube[frameIndex, ...].copy()
        cropIntFrameDn = _.astype('float')
        cropIntFrameDn[overlapRect[0][1]:overlapRect[1][1], overlapRect[0][0]:overlapRect[2][0]] = cropIntFrame_seg_dn[..., 0]
        # ax1['b'].clear();  ax1['b'].imshow(cropIntFrameDn, cmap='gray')
        frameMask = cropIntFrameDn > intThreshold
        # Display masked RGB image: keep pixels where mask is True, black elsewhere
        masked_rgb = rawDat_enfaceSlice.copy()
        if masked_rgb.ndim == 3 and masked_rgb.shape[2] >= 3:
            masked_rgb[~frameMask] = 0
            ax1['c'].clear();  ax1['c'].imshow(masked_rgb)
        else:
            ax1['c'].clear();  ax1['c'].imshow(frameMask, cmap='gray')  # Fallback to show mask in grayscale if input is not RGB

        #todo: apply `frameMask` to rawLIV en-face image
        if VolFlip:  rawLivFrame = tifffile.imread(rawLivFilePath, key=frameIndex)
        else:  rawLivFrame = memmap_rawLiv[:, frameIndex, :]
        rawLivFrame_mask = rawLivFrame * frameMask
        rawLivFrame_mask[rawLivFrame_mask == 0] = np.nan
        # ax1['c'].clear();  ax1['c'].imshow(rawLivFrame_mask, cmap='gray')
        cntLiving = np.sum(rawLivFrame_mask > viabilityThreshold)  # print('Living count is: ', str(cntLiving))
        cntDead = np.sum(rawLivFrame_mask < viabilityThreshold)  # print('Dead count is: ', str(cntDead))
        cntAllPix = np.count_nonzero(~np.isnan(rawLivFrame_mask))  # print('All pixel number is: ', str(cntAllPix))
        viability = cntLiving / (cntLiving + cntDead)  # print('Residual missed count is: ', str(cntAllPix - cntDead - cntLiving))

        # # # # * * * * * * * BULLSHIT func: compute mean logInt (maybe linInt) from the cells  * * * * * * * * #
        # ax1['d'].set_ylim([15, 25]); ax1['d'].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(1.0))
        # rawLivFrame_mask = logIntFrame * frameMask
        # rawLivFrame_mask_norm = rawLivFrame_mask / 255 * 35  # log int, where 0 (correspond to -15dB at measurement) is noise floor
        # rawLivFrame_mask_norm[rawLivFrame_mask_norm == 0] = np.nan
        # # # # Bullshit 1: mean intensity
        # meanInt_mask = np.nanmean(rawLivFrame_mask_norm)
        # viability = meanInt_mask  # mean log intensity [0dB, 35dB] (correspond to [-15dB, 20dB])
        # # # # BULLSHIT 2: percentage based on intensity threshold # # # # #
        # # bullshit_cntLiving = np.sum(rawLivFrame_mask_norm > viaIntThreshold)
        # # bullshit_cntDead = np.sum(rawLivFrame_mask_norm < viaIntThreshold)
        # # cntAllPix = np.count_nonzero(~np.isnan(rawLivFrame_mask_norm))
        # # viability = bullshit_cntLiving / (bullshit_cntLiving + bullshit_cntDead)
        # # # # Bullshit 3: make histogram from 2D en-face or 3D sub-volume? # # # # #
        # # _ = rawLivFrame_mask_norm.flatten()
        # # tmp = _[~np.isnan(_)]
        # # tmpList = np.append(tmpList, tmp)
        # # # # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

        viabilityList.append(viability)
        viaList.append(cntLiving);  totalList.append(cntAllPix)
        if bivariate_mode:
            meafreqFrame = memmap_meanFreq[:, frameIndex, :]
            meafreqFrame_mask = meafreqFrame * frameMask
            meafreqFrame_mask[meafreqFrame_mask == 0] = np.nan
            ax1['c'].clear();  ax1['c'].imshow(meafreqFrame_mask, cmap='hot')
        else:
            ax1['d'].scatter((frameIndex - zSliceList[0]) * 2, viability, color='#6ea6db', marker='o', s=7)
            manual_pick = False

        #todo: Scatter plot of all masked pixels (sparse factor to reduce drawing burden) using (modified) LIV and mean frequency
        if (manual_pick is False) and (bivariate_mode is True):
            y_Liv = rawLivFrame_mask.flatten()
            x_meanFreq = meafreqFrame_mask.flatten() # - 0.7
            valid_mask = ~np.isnan(y_Liv) & ~np.isnan(x_meanFreq)
            # # # (1) 所有mask点绘制 scatter，使用mpl_scatter_density
            white_viridis = LinearSegmentedColormap.from_list('white_viridis', [(0, '#ffffff'), (1e-20, '#440053'),
                (0.2, '#404388'), (0.4, '#2a788e'),(0.6, '#21a784'), (0.8, '#78d151'),(1, '#fde624'),], N=256)
            density = ax1['d'].scatter_density(x_meanFreq[valid_mask], y_Liv[valid_mask], cmap=white_viridis)
            ax1['d'].set_ylim([0, 1]);   # ax1['d'].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], [0, 1, 2, 3, 4, 5, 6, 7])
            ax1['d'].set_xlim([0.5, 8]);  # ax1['d'].set_xscale('log')
            try:  cb.remove()
            except NameError:  pass
            cb = fig1.colorbar(density, ax=ax1['d'], label='Scatter density')
            # try: cb.update_normal(density)  # fail to update colorbar automatically unless manually stretch window
            # except NameError: cb = plt.colorbar(density, ax=ax1['d'], label='Scatter density')

            # # # - - - choose live or dead scatters to plot in ax1['d'] - - - # # #
            if plt.fignum_exists(2): pass
            else:
                fig2 = plt.figure(2, figsize=(4, 4))
                ax2 = fig2.subplot_mosaic("111", per_subplot_kw={"1": dict(projection='scatter_density')})
                ax2['1'].set_xlabel('Mean frequency (Hz)');
                ax2['1'].set_ylabel('LIV (dB$^2$)')
                ax2['1'].set_ylim([0, 1]);  # ax2['1'].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 1, 2, 3, 4, 5])
                ax2['1'].set_xlim([0.5, 6]);  # ax2['1'].set_xscale('log')

            # x_meanFreq_live = x_meanFreq[valid_mask];  y_Liv_live = y_Liv[valid_mask]
            # density_live = ax2['1'].scatter_density(x_meanFreq_live, y_Liv_live, color='green')
            # # # Or
            x_meanFreq_dead = x_meanFreq[valid_mask];  y_Liv_dead = y_Liv[valid_mask]
            density_dead = ax2['1'].scatter_density(x_meanFreq_dead, y_Liv_dead, color='red')


            # # # 最后一个en-face画histogram # # #
            if frameIndex != zSliceList[-1]:  pass;
            else:
                metric_FreqLiv = y_Liv[valid_mask] * 5  # np.multiply(x_meanFreq[valid_mask], y_Liv[valid_mask] * 5)  # re-scale from m-LIV to LIV (rough)
                hist_FreqLiv, bins_FreqLiv = np.histogram(metric_FreqLiv, bins=50, density=True)
                if plt.fignum_exists(2):  pass
                else:
                    fig2 = plt.figure(figsize=(4, 4))
                    ax2 = fig2.add_subplot(111)
                ax2.hist(metric_FreqLiv, bins=bins_FreqLiv, density=True,alpha=0.4)
                # bin_width = bins_FreqLiv[1] - bins_FreqLiv[0]
                # percentage = np.sum(bin_width * hist_FreqLiv[np.where(bins_FreqLiv[:-1] < 0.91)])  # LIV*f_mean: 1.8; LIV: 0.9

        #todo: Manual pick for scatter plot
        elif manual_pick:
            print(f"Frame {frameIndex}: Manual labeling. Left-click to mark 'livng' as green; right-click to mark 'dead' as red. 'Enter' to complete.")
            pts_lime.clear();  pts_magenta.clear()
            picking = True
            def on_m_click(event):
                if event.inaxes == ax1['a']:
                    if event.button is MouseButton.LEFT:
                        pts_lime.append((int(event.xdata), int(event.ydata)))
                        ax1['a'].plot(event.xdata, event.ydata, 'g+')
                    elif event.button is MouseButton.RIGHT:
                        pts_magenta.append((int(event.xdata), int(event.ydata)))
                        ax1['a'].plot(event.xdata, event.ydata, 'm+')
                    fig1.canvas.draw()
            def on_m_key(event):
                global picking, manual_pick
                if event.key == 'enter':
                    picking = False
                    manual_pick = None
            def on_close(event):
                global picking, manual_pick
                picking = False
                manual_pick = None
            cid_m_c = fig1.canvas.mpl_connect('button_press_event', on_m_click)
            cid_m_k = fig1.canvas.mpl_connect('key_press_event', on_m_key)
            cid_close = fig1.canvas.mpl_connect('close_event', on_close)
            while picking: plt.pause(0.5)
            fig1.canvas.mpl_disconnect(cid_m_c);  fig1.canvas.mpl_disconnect(cid_m_k)
            for pts, color, marker, kwargs in [(pts_lime, 'lime', 'o', {'edgecolors': 'k'}), (pts_magenta, 'magenta', 'x', {'linewidths': 2})]:
                if pts:
                    m_pts = np.array(pts)
                    m_y, m_x = m_pts[:, 1], m_pts[:, 0]
                    valid_m = (m_y >= 0) & (m_y < rawLivFrame.shape[0]) & (m_x >= 0) & (m_x < rawLivFrame.shape[1])
                    m_y, m_x = m_y[valid_m], m_x[valid_m]
                    ax1['d'].scatter(meafreqFrame[m_y, m_x], rawLivFrame[m_y, m_x], s=40, c=color, marker=marker, zorder=20, **kwargs)
            fig1.canvas.draw()

    except IndexError:
        pass

    # sys.stdout.write('\r')
    # j = (frameIndex - zSliceList[0] + 1) / len(zSliceList)
    # sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j) + ' couting viability over Zstack')
    plt.pause(0.1)

print('Mean en-face fraction (vFrac) is: ', str(np.mean(viabilityList)))
print('std is: ', str(np.std(viabilityList)))
print('Volume fraction (vFrac\') is: ', str(np.sum(viaList) / np.sum(totalList)))
# # # Bullshit 3: make histogram from 2D en-face or 3D sub-volume? # # # # #
# # # counts, bin_edges = np.histogram(tmpList, bins=20);
# # # k_mean = np.mean(counts); k_var = np.var(counts);
# # # k_c = (2 * k_mean - k_var) / (bin_edges[2] - bin_edges[1])**2
# # # * Above: Shimazaki & Shinomoto bin width optimization, find bins= that minimize k_c * # # #
# plt.figure(21);
# plt.hist(tmpList, facecolor='red', bins=30, range=[0, 30], alpha=0.35, density=False)  # Dead
# plt.hist(tmpList, facecolor='green', bins=30, range=[0, 30], alpha=0.35, density=False)  # Live
# # # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
