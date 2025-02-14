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
import imagej
import tifffile
import os
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
# from mpl_point_clicker import clicker
from matplotlib.backend_bases import MouseButton
import matplotlib.patches as patches
from cellpose import denoise #, utils, io
import matplotlib
matplotlib.use("Qt5Agg")


"""
# Open source project for cell counting system. 
# Compute viability based on the normalized LIV by calculating the volume fraction above and under a given threshold. 
# Author: Yijie, Lida
# 
# Parameters: 
# zSlice: depth range of Z-stack images for computing viability
# intThreshold: normalized log intensity threshold to mask out cell regions. Note that the log intensity image (en-face) is the maximum log intensity projection along each raster period, then being denoised by Cellpose v3 using 'cyto3' model.
# viabilityThreshold: LIV threshold to determine whether the LIV value in a pixel represents Living / dead. 
# 
# How to use:
# 1. run ImageConverter.py, to convert raw data (log int) to linear and log int image stacks (Data_IntImg.tif and Data_3d_view.tif). 
# 2. run VarianceToRGB.py, to encode temporal variance of log int as Hue, max log int (during raster period) as Value, 1 as Saturation. 
# 3. open Data_IntImg_LIV.tif in ImageJ, and measure the tilting angle along X (Bscan_tilt) and Y (Y_tilt)
# 4. run /Fiji_macro/AutoRotateMacro.ijm, manually set 'Bscan_tilt' and 'Y_tilt' from the above measurements, and process all 3 image stacks. 
# 5. open aligned LIV image (Data_IntImg_LIV.tif), and select the depth range for computing viability. 
# 6. run VarianceToViability.py, manually set 'zSlice' to be the determined depth range, and select the LIV image (Data_IntImg_LIV.tif) to start computing viability. 
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
        plt.gca().add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='y', facecolor='none'))
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


zSlice = [530, 560]  # manual z slicing range to select depth region for computing viability
intThreshold = 0.3
viabilityThreshold = 0.2

tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True); stackFilePath = filedialog.askopenfilename(filetypes=[("", "*_LIV.tif")])
DataId = os.path.basename(stackFilePath);   root = os.path.dirname(stackFilePath);  tk.destroy()
print('Loading fileID: ' + stackFilePath)

# # # # - - - read size of tiff stack - - - # # #
rawDat = tifffile.imread(stackFilePath)   # load linear intensity data from stack. Dimension (Y, Z, X)
dim_z, dim_y, dim_x = np.shape(rawDat)[0:3]  # [dim_y, dim_z, dim_x] original dimensions before applying AutoRotateMacro.ijm
cropCube = np.zeros([dim_z, dim_y, dim_x], dtype=int)
model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")
zSliceList = np.linspace(zSlice[0], zSlice[1], zSlice[1]-zSlice[0]+1).astype('int')

fig1 = plt.figure(10, figsize=(12, 5));  plt.clf()
# # fig1.canvas.manager.window.attributes('-topmost', 1);  fig1.canvas.manager.window.attributes('-topmost', 0)
# # fig1.subplots_adjust(bottom=0, top=1, left=0, right=1)
ax1 = fig1.subplot_mosaic("abc;abc;ddd")
ax1['a'].title.set_text('Drag rectangle to select ROI from dOCT')
ax1['b'].title.set_text('After manual cropping')
ax1['c'].title.set_text('Segmentation mask')
ax1['d'].set_ylabel('Viable fraction');  ax1['d'].set_xlabel('En-face slice at depth (um)')
ax1['d'].set_ylim([-0.01, 1.02]);  ax1['d'].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax1['d'].set_xlim([0, len(zSliceList)*2]);

# # # draw rectangles at the first and last frames, and the overlapping cubic is the 3D ROI for viability (volume fraction) computation
frameIndex = zSliceList[0]
firstFrameCord = drawRectFromFrame(ax1['a'], fig1, rawDat, frameIndex)
frameIndex = zSliceList[-1]
lastFrameCord = drawRectFromFrame(ax1['a'], fig1, rawDat, frameIndex)

# # # create 3D mask from the two rectangles' coordinates
overlapRect = findOverlapRect(firstFrameCord, lastFrameCord)
# overlapRect = findOverlapRect([(2, 1), (2, 237), (254, 1), (254, 237)], [(3, 3), (3, 197), (251, 3), (251, 197)])
cropCube[:, overlapRect[0][1]:overlapRect[1][1], overlapRect[0][0]:overlapRect[2][0]] = 1  # dim_z, dim_y, dim_x

# # load linear intensity stack, apply cropCube, denoising using Cellpose v3, apply intThreshold to segment cell regions
# # 3D version is only for test visualize, a 2D version is preferred which is supposed to work with an en-face image stack
linIntFilePath = root + '/' + DataId[:-15] + '_3d_view.tif'
rawLivFilePath = root + '/' + DataId[:-15] + '_IntImg_LIV_raw.tif'


viabilityList = [];  viaList = [];  totalList = []
for frameIndex in zSliceList:
    logIntFrame = tifffile.imread(linIntFilePath, key=frameIndex)
    # ax1['a'].clear();  ax1['a'].imshow(logIntFrame, cmap='gray')
    ax1['a'].clear();  ax1['a'].imshow(rawDat[frameIndex, ...])  # cropIntFrame = logIntFrame.copy() * cropCube[frameIndex, ...]
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
        ax1['c'].clear();  ax1['c'].imshow(frameMask, cmap='gray')

        # # # apply frameMask to rawLIV en-face image
        rawLivFrame = tifffile.imread(rawLivFilePath, key=frameIndex)
        rawLivFrame_mask = rawLivFrame * frameMask
        rawLivFrame_mask[rawLivFrame_mask == 0] = np.nan
        # ax1['c'].clear();  ax1['c'].imshow(rawLivFrame_mask, cmap='gray')
        cntLiving = np.sum(rawLivFrame_mask > viabilityThreshold)  # print('Living count is: ', str(cntLiving))
        cntDead = np.sum(rawLivFrame_mask < viabilityThreshold)  # print('Dead count is: ', str(cntDead))
        cntAllPix = np.count_nonzero(~np.isnan(rawLivFrame_mask))  # print('All pixel number is: ', str(cntAllPix))
        viability = cntLiving / (cntLiving + cntDead)  # print('Residual missed count is: ', str(cntAllPix - cntDead - cntLiving))
        # * * * * * * * BULLSHIT func: compute mean logInt (maybe linInt) from the cells  * * * * * * * * #
        # rawLivFrame_mask = logIntFrame * frameMask
        # rawLivFrame_mask_norm = rawLivFrame_mask / 255
        # rawLivFrame_mask_norm[rawLivFrame_mask_norm == 0] = np.nan
        # meanInt_mask = np.nanmean(rawLivFrame_mask_norm)
        # viability = meanInt_mask
        # * * * * * * * BULLSHIT func: compute mean logInt (maybe linInt) from the cells  * * * * * * * * #
        viabilityList.append(viability);  viaList.append(cntLiving);  totalList.append(cntAllPix)
        ax1['d'].scatter((frameIndex-zSliceList[0])*2, viability, color='#6ea6db', marker='o', s=7)  # , np.mean(viabilityList)
    except IndexError:
        pass

    # sys.stdout.write('\r')
    # j = (frameIndex - zSliceList[0] + 1) / len(zSliceList)
    # sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j) + ' couting viability over Zstack')
    plt.pause(0.01)

print('Mean en-face fraction (vFrac) is: ', str(np.mean(viabilityList)))
print('std is: ', str(np.std(viabilityList)))
print('Volume fraction (vFrac\') is: ', str(np.sum(viaList) / np.sum(totalList)))
