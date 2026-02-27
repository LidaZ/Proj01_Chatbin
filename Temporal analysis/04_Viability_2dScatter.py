# import os
# import sys
# import time
import numpy as np
import tifffile
import os
import glob
from tkinter import *
from tkinter import filedialog
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpmath.libmp.libelefun import log_int_cache
# from torch.utils.tensorboard.summary import histogram
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
# from mpl_point_clicker import clicker
from matplotlib.backend_bases import MouseButton
import matplotlib.patches as patches
from cellpose import denoise #, utils, io
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable



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
# 6. run 04_Viability_2dScatter.py, manually set 'zSlice' to be the determined depth range, and select the LIV image (Data_IntImg_LIV.tif) to start computing viability. 
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


# x_meanFreq_tmp = []
# y_Liv_tmp = []


zSlice = [338, 339]  # manual z slicing range to select depth region for computing viability
int_threshold_segment = 0.35  # Segmentation after cellpose-3 process, which is a normalized image.
viability_liv_threshold = 0.18
Switch_Sscatter2D_or_mLivCountViability = True  # True: scatter plot; False: pixel counting for viability
second_metric_horizontal = 'swiftness'  # 'swiftness' or 'mean frequency'
# viaIntThreshold = 13  # bullshit threshold on intensity to compute viability
manual_pick = False  # Enable manual pixel labeling on ax1['a']. Automatic display all masked pixels when set to False.
aliv_range = (0, 35)
mliv_range = (0, 1)  # liv_range = (0, 6)
swiftness_range = (0, 50)
meanFreq_range = (0, 6)


tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True)
stackFilePath = filedialog.askopenfilename(title="aliv > YasunoAliv&Swift; LIV > mLiv", filetypes=[("", "*aliv_min* *_LIV.tif")])
DataId = os.path.basename(stackFilePath);   root = os.path.dirname(stackFilePath);  tk.destroy()
first_metric_aLivOvermLiv = True if 'aliv' in DataId else False  # True: aLIV + swiftness; False: mean frequency
print('Loading fileID: ' + stackFilePath)

""" Read size of tiff stack. If VolFlip==True:  rawDat = tifffile.imread(stackFilePath) """
memmap_rawDat = tifffile.memmap(stackFilePath, mode='r')
rawDat = np.swapaxes(memmap_rawDat, 0, 1)  # load linear intensity data from stack. Dimension (Y, Z, X)
dim_z, dim_y, dim_x = np.shape(rawDat)[0:3]            # [dim_y, dim_z, dim_x] original dimensions before applying `AutoRotateMacro.ijm`
cropCube = np.zeros([dim_z, dim_y, dim_x], dtype=int)
model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")
try:  zSliceList = np.linspace(zSlice[0], zSlice[1], zSlice[1]-zSlice[0]+1).astype('int')
except ValueError:  print('zSlice is not a valid range: ', zSlice)

# if plt.fignum_exists(1): plt.close(1)
    # fig1 = plt.figure(1)  # plt.clf()  # pass
    # ax1 = {label: ax for ax, label in zip(fig1.axes, ['a', 'b', 'c', 'd'])}  # 获取已存在的 axes
fig1 = plt.figure(1, clear=True, figsize=(9, 4.5), layout='constrained')
ax1 = fig1.subplot_mosaic("aaabc;aaadd;aaadd", per_subplot_kw={"d": dict(projection='scatter_density')})
ax1['a'].title.set_text('Drag rectangle to select ROI from dOCT')
ax1['b'].title.set_text('After manual cropping'); ax1['b'].axis('off')
ax1['c'].title.set_text('Segmentation mask'); ax1['c'].axis('off')
if Switch_Sscatter2D_or_mLivCountViability is False:
    ax1['d'].set_ylabel('Viable fraction');  ax1['d'].set_xlabel('En-face slice at depth (um)')
    ax1['d'].set_ylim(-0.01, 1.02);  ax1['d'].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1['d'].set_xlim(0, len(zSliceList)*2)
else:
    ax1['d'].set_ylabel('aLIV (dB$^2$)') if first_metric_aLivOvermLiv else ax1['d'].set_ylabel('mLIV (dB)')
    ax1['d'].set_xlabel('Swiftness (s$^{-1}$)') if second_metric_horizontal == 'swiftness' else ax1['d'].set_xlabel('Mean frequency (Hz)')

""" Draw rectangles at the first and last frames, and the overlapping cubic is the 3D ROI for viability (volume fraction) computation. """
frameIndex = zSliceList[0]
firstFrameCord = drawRectFromFrame(ax1['a'], fig1, rawDat, frameIndex)
frameIndex = zSliceList[-1]
lastFrameCord = drawRectFromFrame(ax1['a'], fig1, rawDat, frameIndex)

""" Create 3D mask from the two rectangles' coordinates. """
overlapRect = findOverlapRect(firstFrameCord, lastFrameCord)
# overlapRect = findOverlapRect([(2, 1), (2, 237), (254, 1), (254, 237)], [(3, 3), (3, 197), (251, 3), (251, 197)])
cropCube[:, overlapRect[0][1]:overlapRect[1][1], overlapRect[0][0]:overlapRect[2][0]] = 1  # dim_z, dim_y, dim_x

""" load linear intensity stack, apply `cropCube`, denoising using `Cellpose v3`, apply `intThreshold` to segment cell regions. 
3D version is only for test visualize, a 2D version is preferred which is supposed to work with an en-face image stack. """
string_DataId = DataId[:4] # if first_metric_aLivOvermLiv else DataId[:-15]
logIntFilePath = root + '/' + string_DataId + '_3d_view.tif'
if first_metric_aLivOvermLiv:  first_metricLIV_FilePath = glob.glob(root + '/' + string_DataId + '_IntImg_aliv.tif')[0]
else:  first_metricLIV_FilePath = root + '/' + string_DataId + '_IntImg_LIV_raw.tif'
if second_metric_horizontal == 'swiftness':  second_metric_filePath = root + '/' + string_DataId + '_IntImg_swiftness.tif'
elif second_metric_horizontal == 'mean frequency':  second_metric_filePath = root + '/' + string_DataId + '_IntImg_meanFreq.tif'

viabilityList = [];  viaList = [];  totalList = []
tmpList = []
pts_lime = [];  pts_magenta = []
memmap_logInt = tifffile.memmap(logIntFilePath, mode='r')
memmap_rawLiv = tifffile.memmap(first_metricLIV_FilePath, mode='r')
if Switch_Sscatter2D_or_mLivCountViability:
    try:  memmap_second_metric = tifffile.memmap(second_metric_filePath, mode='c')
    except FileNotFoundError:  print('Scatter plot mode, did not find second-metric data: ', second_metric_filePath)

for frameIndex in zSliceList:
    logIntFrame = memmap_logInt[:, frameIndex, :]
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
        frameMask = cropIntFrameDn > int_threshold_segment
        """ Display masked RGB image: keep pixels where mask is True, black elsewhere. """
        masked_rgb = rawDat_enfaceSlice.copy()
        if masked_rgb.ndim == 3 and masked_rgb.shape[2] >= 3:
            masked_rgb[~frameMask] = 0
            ax1['c'].clear();  ax1['c'].imshow(masked_rgb)
        else:
            ax1['c'].clear();  ax1['c'].imshow(frameMask, cmap='gray')  # Fallback to show mask in grayscale if input is not RGB

        """ Apply `frameMask` to rawLIV en-face image. """
        rawLivFrame = memmap_rawLiv[:, frameIndex, :]  # if VolFlip:  rawLivFrame = tifffile.imread(rawLivFilePath, key=frameIndex).
        rawLivFrame_mask = rawLivFrame * frameMask
        rawLivFrame_mask[rawLivFrame_mask == 0] = np.nan
        # ax1['c'].clear();  ax1['c'].imshow(rawLivFrame_mask, cmap='gray')
        cntLiving = np.nansum(rawLivFrame_mask > viability_liv_threshold)  # print('Living count is: ', str(cntLiving))
        cntDead = np.nansum(rawLivFrame_mask < viability_liv_threshold)  # print('Dead count is: ', str(cntDead))
        cntAllPix = cntDead + cntLiving # np.count_nonzero(~np.isnan(rawLivFrame_mask))  # print('All pixel number is: ', str(cntAllPix))
        viability = cntLiving / cntAllPix  # print('Residual missed count is: ', str(cntAllPix - cntDead - cntLiving))

        # # # # * * * * * * * BULLSHIT func: compute mean logInt (maybe linInt) from the cells  * * * * * * * * #
        # ax1['d'].set_ylim(15, 25); ax1['d'].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(1.0))
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
        if Switch_Sscatter2D_or_mLivCountViability:
            second_metric_frame = memmap_second_metric[:, frameIndex, :]
            # # #todo: 把拟合的swiftness中的所有infinity强制设为0. 需要看看为啥gpufit得到的1/tau是无限
            second_metric_frame[second_metric_frame == np.inf] = 0
            second_metric_mask = second_metric_frame * frameMask
            second_metric_mask[second_metric_mask == 0] = np.nan
            ax1['c'].clear(); ax1['c'].imshow(second_metric_mask, cmap='inferno')
        else:
            ax1['d'].scatter((frameIndex - zSliceList[0]) * 2, viability, color='#6ea6db', marker='o', s=7)
            manual_pick = False

        """ Scatter plot of all masked pixels (mpl_scatter_density to reduce drawing burden) using (modified) LIV and mean frequency. """
        if (manual_pick is False) and (Switch_Sscatter2D_or_mLivCountViability is True):
            y_first_metric = rawLivFrame_mask.flatten()
            x_second_metric = second_metric_mask.flatten()
            valid_mask = ~np.isnan(y_first_metric) & ~np.isnan(x_second_metric)  # np.isinf(valid_mask).any() > False
            # # # (1) 所有mask点绘制 scatter，使用mpl_scatter_density
            # white_viridis = LinearSegmentedColormap.from_list('white_viridis', [(0, '#ffffff'), (1e-20, '#440053'),
            #     (0.2, '#404388'), (0.4, '#2a788e'),(0.6, '#21a784'), (0.8, '#78d151'),(1, '#fde624'),], N=256)
            self_cmap = LinearSegmentedColormap.from_list('self_cmap', [(0, '#ffffff'), (1e-20, '#160c57'),
                (0.2, '#2469fd'), (0.4, '#24fdfd'), (0.6, '#94fd24'), (0.8, '#fdce24'), (1, '#fd2f24'), ], N=256)
            density = ax1['d'].scatter_density(x_second_metric[valid_mask], y_first_metric[valid_mask], cmap=self_cmap)
            try:  cb.remove()
            except NameError:  pass
            cb = fig1.colorbar(density, ax=ax1['d'], label='Scatter density')
            if first_metric_aLivOvermLiv:  ax1['d'].set_ylabel('aLIV (dB$^2$)'); ax1['d'].set_ylim(aliv_range)
            else:  ax1['d'].set_ylabel('mLIV (dB)'); ax1['d'].set_ylim(mliv_range)
            if second_metric_horizontal == 'swiftness':  ax1['d'].set_xlim(swiftness_range)
            elif second_metric_horizontal == 'mean frequency': ax1['d'].set_xlim(meanFreq_range)
                # ax1['d'].set_ylabel('LIV (dB$^2$)'); ax1['d'].set_ylim(0, 7)
                # ax1['d'].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 1, 2, 3, 4, 5])  # rescale from mLIV (dB) to LIV (dB^2)
            # try: cb.update_normal(density)  # fail to update colorbar automatically unless manually stretch window
            # except NameError: cb = plt.colorbar(density, ax=ax1['d'], label='Scatter density')

            """ Use the last en-face for scatter plot with histogram - Figure 2 """
            # if frameIndex == zSliceList[-1]:
            #     if not plt.fignum_exists(2):
            #         fig2 = plt.figure(2, figsize=(4, 4))
            #         ax2 = fig2.subplot_mosaic("111", per_subplot_kw={"1": dict(projection='scatter_density')})
            #         if switch_to_yasuno_aLiv_swiftness:
            #             ax2['1'].set_xlabel('Swiftness (s$^-1$)');  ax2['1'].set_ylabel('aLIV (dB$^2$)')
            #         else:
            #             ax2['1'].set_xlabel('Mean frequency (Hz)');  ax2['1'].set_ylabel('LIV (dB$^2$)')
            #         ax2['1'].set_ylim(mliv_range);  # ax2['1'].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 1, 2, 3, 4, 5])
            #         ax2['1'].set_xlim(meanFreq_range);  # ax2['1'].set_xscale('log')
            #
            #         divider = make_axes_locatable(ax2['1'])
            #         ax_histx = divider.append_axes("top", size="25%", pad=0.01, sharex=ax2['1'])
            #         ax_histx.xaxis.set_tick_params(labelbottom=False);  ax_histx.xaxis.set_visible(False);  ax_histx.axis('off')
            #         ax_histy = divider.append_axes("right", size="25%", pad=0.01, sharey=ax2['1'])
            #         ax_histy.yaxis.set_tick_params(labelleft=False);  ax_histy.yaxis.set_visible(False);  ax_histy.axis('off')
            #     """ assign color. """
            #     scatter_color = 'red'
            #     x_meanFreq_masked = x_second_metric[valid_mask];     y_Liv_masked = y_Liv[valid_mask]
            #     density_dead = ax2['1'].scatter_density(x_meanFreq_masked, y_Liv_masked, color=scatter_color)
            #     histx, bins_to_histx = np.histogram(x_meanFreq_masked, bins=50, density=True)
            #     ax_histx.hist(x_meanFreq_masked, bins=bins_to_histx, density=True, alpha=0.4, color=scatter_color)
            #     histy, bins_to_histy = np.histogram(y_Liv_masked, bins=50, density=True)
            #     ax_histy.hist(y_Liv_masked, bins=bins_to_histy, density=True, alpha=0.4, color=scatter_color, orientation='horizontal')


        # #todo: Manual pick for scatter plot. 2026/02/26: Can be removed?
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
                    ax1['d'].scatter(second_metric_frame[m_y, m_x], rawLivFrame[m_y, m_x], s=40, c=color, marker=marker, zorder=20, **kwargs)
            fig1.canvas.draw()

    except IndexError:  pass

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
