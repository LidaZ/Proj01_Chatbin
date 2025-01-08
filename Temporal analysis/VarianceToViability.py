import os
import sys
import time
import numpy as np
from numpy import fft
import gc
import matplotlib.cm as cm
# from tqdm.auto import tqdm
# import pandas as pd
# from scipy.stats import norm
# from scipy.optimize import curve_fit
from lowpass_filter import butter_lowpass_filter
from MapVarToRGB import num_to_rgb
from matplotlib.colors import hsv_to_rgb
import tifffile
import os
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
# from mpl_point_clicker import clicker
from matplotlib.backend_bases import MouseButton
import matplotlib.patches as patches
# import matplotlib
# matplotlib.use("Qt5Agg")


"""
Open source project for cell counting system. 
Compute viability based on the normalized LIV by calculating the volume fraction above and under a given threshold. 
Author: Yijie, Lida

Parameters: 

How to use:

Para setting: 

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
    LeftUp = (max(firstFrameCord[0][0], lastFrameCord[0][0]), min(firstFrameCord[0][1], lastFrameCord[0][1]))
    LeftDown = (LeftUp[0], min(firstFrameCord[1][1], lastFrameCord[1][1]))
    RightUp = (min(firstFrameCord[2][0], lastFrameCord[2][0]), min(firstFrameCord[2][1], lastFrameCord[2][1]))
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
    FrameCord = rectangle_coords; print('selected frame: ', str(frameId), ' ROI is: ', FrameCord)
    return FrameCord


# intThreshold = ?
viabilityThreshold = 0.3
zSlice = [403, 463]

tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True); stackFilePath = filedialog.askopenfilename(filetypes=[("", "*_LIV.tif")])
DataId = os.path.basename(stackFilePath);   root = os.path.dirname(stackFilePath);  tk.destroy()
if '_LIV' in DataId:  pass
else:  raise(ValueError('Select variance image for better visualization'))
print('Loading data folder: ' + root)

# # # - - - read size of tiff stack - - - # # #
rawDat = tifffile.imread(stackFilePath)   # load linear intensity data from stack. Dimension (Y, Z, X)
dim_y, dim_z, dim_x = np.shape(rawDat)[0:3]
maskCube = np.zeros([dim_y, dim_z, dim_x], dtype=int)

fig1 = plt.figure(10, figsize=(3, dim_z/dim_x*3));  plt.clf()
# fig1.canvas.manager.window.attributes('-topmost', 1);  fig1.canvas.manager.window.attributes('-topmost', 0)
fig1.subplots_adjust(bottom=0, top=1, left=0, right=1)
ax1 = fig1.subplot_mosaic("a")

# # # draw rectangles at the first and last frames, and the overlapping cubic is the 3D ROI for viability (volume fraction) computation
# _ = ax1['a'].imshow(rawDat[-1, :, :, :])
# start_point = None;  end_point = None;  rect = None;  selectNotDone = True;  rectangle_coords = []
# fig1.canvas.mpl_connect('button_press_event', on_press)
# fig1.canvas.mpl_connect('motion_notify_event', on_drag)
# roi_lastFrame = fig1.canvas.mpl_connect('button_release_event', on_release)
# while selectNotDone:  plt.pause(0.3)
# lastFrameCord = rectangle_coords;  print('second frame ROI is: ', lastFrameCord)
firstFrameCord = drawRectFromFrame(ax1['a'], fig1, rawDat, 0)
lastFrameCord = drawRectFromFrame(ax1['a'], fig1, rawDat, -1)

# # # create 3D mask from the two rectangles' coordinates
overlapRect = findOverlapRect(firstFrameCord, lastFrameCord)
maskCube[overlapRect[0][1]:overlapRect[1][1], :, overlapRect[0][0]:overlapRect[2][0]] = 1

# # # load linear intensity stack, and apply intThreshold + maskCube to segment cell regions
frameIndex = -1
linIntFilePath = root + '/' + DataId[:-14] + 'IntImg.tif'
tmp_linIntFrame = tifffile.imread(linIntFilePath, key=frameIndex)
log_tmp = 10 * np.log10(tmp_linIntFrame)

fig2 = plt.figure(11); plt.clf()
ax2 = fig2.subplot_mosaic("abcd")
ax2['a'].imshow(rawDat[0, ..., 0], cmap='gray')
ax2['b'].imshow(rawDat[0, ..., 1], cmap='gray')
ax2['c'].imshow(rawDat[0, ..., 2], cmap='gray')
ax2['d'].imshow(log_tmp, cmap='gray')

