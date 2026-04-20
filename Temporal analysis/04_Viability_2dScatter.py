import numpy as np
import tifffile
import os
import glob
from tkinter import *
from tkinter import filedialog
import matplotlib
# from fontTools.unicodedata import block
# from astropy.visualization import LogStretch
# from astropy.visualization.mpl_normalize import ImageNormalize
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
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

def drawRectFromFrame(ax1, fig1, memmap_logInt_reshape_YZX, frameId):
    global start_point, end_point, rect, rectangle_coords, selectNotDone
    _ = ax1.imshow(memmap_logInt_reshape_YZX[frameId, ...])  # memmap_logInt_reshape_YZX[frameId, dim_z, dim_x, ch]
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

def convert_arr_grayscale_to_dB(intArray_grayscale, octRangedB):
    intArray_db = intArray_grayscale / 255 * (octRangedB[1] - octRangedB[0]) + octRangedB[0]
    return intArray_db

def config_metrics_folderpath(string_DataId, first_metric_horizontal, second_metric_horizontal):
    """ Define metrics folder path. """
    global first_metric_label, first_metricLIV_FilePath, first_metric_range, PATH_metric_loaded_forDisplay, second_metric_label, second_metric_filePath, second_metric_range
    match first_metric_horizontal:
        case 'aliv':
            first_metric_label = 'aLIV (dB$^2$)'
            first_metricLIV_FilePath = glob.glob(root + '/' + string_DataId + '_IntImg_aliv.tif')[0]
            first_metric_range = aliv_range
            PATH_metric_loaded_forDisplay = glob.glob(root + '/' + string_DataId + '*aliv_min*.tif')[0]
        case 'liv':
            first_metric_label = 'LIV (dB$^2$)'
            first_metricLIV_FilePath = root + '/' + string_DataId + '_IntImg_LIV_raw.tif'
            first_metric_range = liv_range
            PATH_metric_loaded_forDisplay = root + '/' + string_DataId + '_IntImg_LIV.tif'
        case 'mliv':
            first_metric_label = 'mLIV (dB)'
            first_metricLIV_FilePath = glob.glob(root + '/' + string_DataId + '_IntImg_mliv_raw.tif')[0]
            first_metric_range = mliv_range
            PATH_metric_loaded_forDisplay = root + '/' + string_DataId + '_IntImg_mLIV.tif'
        case _:
            print('Error: first metric is not available: ', DataId)

    match second_metric_horizontal:
        case 'swiftness':
            second_metric_label = 'Swiftness (s$^{-1}$)'
            second_metric_filePath = root + '/' + string_DataId + '_IntImg_swiftness.tif'
            second_metric_range = swiftness_range
        case 'mean frequency':
            second_metric_label = 'Mean frequency (Hz)'
            second_metric_filePath = root + '/' + string_DataId + '_IntImg_meanFreq.tif'
            second_metric_range = meanFreq_range
        case 'mean intensity':
            second_metric_label = 'Mean intensity (dB)'
            second_metric_filePath = root + '/' + string_DataId + '_3d_view.tif'
            second_metric_range = octRangedB_display
        case _:
            print('Error: second metric is not available: ', second_metric_horizontal)

def load_and_display_enface_image(mLivFilePath, ax1):
    memmap_mliv = tifffile.memmap(mLivFilePath, mode='r')
    ax1.clear(); ax1.imshow(memmap_mliv[:, index_z, ...])



zSlice = [436, 437]  # manual z slicing range to select depth region for computing viability
SYSTEM_ID = 'ivs800'  # 'ivs800' or 'ivs2000'
plot_mode = 'scatter'  # 'counting' or 'scatter'
first_metric_horizontal = 'liv'  # 'aLIV', 'liv', or 'mliv'
second_metric_horizontal = 'mean intensity'  # 'swiftness', 'mean frequency' or 'mean intensity'. Only available when switch_scatter2D == True.

int_threshold_dB = -10.  # Threshold for log intensity to segment cell regions. If cellpose-3 enabled, it's applied to a normalized image, with typical value of ~0.17.
viability_liv_threshold = 0.18  # 'mLIV': 0.18; 'LIV': ?; 'aLIV': ?

octRangedB = (-15, 20) if SYSTEM_ID == 'ivs800' else (0, 50) if SYSTEM_ID == 'ivs2000' else print('Error: system ID is not supported: ', SYSTEM_ID)
octRangedB_display = (int_threshold_dB, octRangedB[1])
aliv_range = (0, 35);  swiftness_range = (0, 50);  exclude_inf_swiftness = True
liv_range = (0, 45)
mliv_range = (0, 1)
meanFreq_range = (0, 6)
manual_pick = False  # Enable manual pixel labeling on ax1['a']. Automatic display all masked pixels when set to False.

""" Open file to select first metric. aliv.tif > aliv, _LIV.tif > mLiv. """
file_name_filter = "*.bin"
tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True)
stackFilePath = filedialog.askopenfilename(filetypes=[("", file_name_filter)])
DataId = os.path.basename(stackFilePath);   root = os.path.dirname(stackFilePath);  tk.destroy()
print('Loading fileID: ' + stackFilePath)

""" Define axes property for 2D scatter plot. """
string_DataId = DataId[:-4]
config_metrics_folderpath(string_DataId, first_metric_horizontal, second_metric_horizontal)

""" Read size of log intensity stack, for drawing ROI box. """
memmap_mliv = tifffile.memmap(PATH_metric_loaded_forDisplay, mode='r')  # memmap_mlivFilePath_reshape_YZX = tifffile.imread(stackFilePath)
memmap_mliv_reshape_YZX = np.swapaxes(memmap_mliv, 0, 1)  # load linear intensity data from stack. Dimension (Y, Z, X)
dim_z, dim_y, dim_x = np.shape(memmap_mliv_reshape_YZX)[0:3]            # [dim_y, dim_z, dim_x] original dimensions before applying `AutoRotateMacro.ijm`
enface_mask_to_vol = np.zeros([dim_z, dim_y, dim_x], dtype=int)
# #todo: temporarily remove cellpose filter, to completely exclude low-intensity pixels.
# model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")
try:  z_range = np.linspace(zSlice[0], zSlice[1], zSlice[1] - zSlice[0] + 1).astype('int')
except ValueError:  print('zSlice is not a valid range: ', zSlice)

fig1 = plt.figure(1, clear=True, figsize=(9, 4.5), layout='constrained')
ax1 = fig1.subplot_mosaic("aaabc;aaadd;aaadd", per_subplot_kw={"d": dict(projection='scatter_density')})
ax1['a'].title.set_text('Drag rectangle to select ROI from dOCT')
ax1['b'].title.set_text('After manual cropping'); ax1['b'].axis('off')
ax1['c'].title.set_text('Segmentation mask'); ax1['c'].axis('off')
if plot_mode == 'counting':
    ax1['d'].set_ylabel('Viable fraction');  ax1['d'].set_xlabel('En-face slice at depth (um)')
    ax1['d'].set_ylim(-0.01, 1.02);  ax1['d'].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1['d'].set_xlim(0, len(z_range) * 2)

""" Select ROI (overlap) for computation. """
firstFrameCord = drawRectFromFrame(ax1['a'], fig1, memmap_mliv_reshape_YZX, z_range[0])
lastFrameCord = drawRectFromFrame(ax1['a'], fig1, memmap_mliv_reshape_YZX, z_range[1])

""" Create 3D mask to segment ROI using rectangles' coordinates. """
overlapRect = findOverlapRect(firstFrameCord, lastFrameCord)
enface_mask_to_vol[:, overlapRect[0][1]:overlapRect[1][1], overlapRect[0][0]:overlapRect[2][0]] = 1  # dim_z, dim_y, dim_x

""" load first and second metrics raw data, then segment using log-intensity-based mask. 
Old version: load linear intensity stack, apply `cropCube`, denoising using `Cellpose v3`, apply `intThreshold` to segment cell regions. 
3D version is only for test visualize, a 2D version is preferred which is supposed to work with an en-face image stack. """
viabilityList = [];  viaList = [];  totalList = []; tmpList = []; pts_lime = [];  pts_magenta = []
memmap_first_metric = tifffile.memmap(first_metricLIV_FilePath, mode='r')
if plot_mode == 'scatter':
    try:  memmap_second_metric = tifffile.memmap(second_metric_filePath, mode='c')
    except FileNotFoundError:  print('Scatter plot mode, did not find second-metric data: ', second_metric_filePath)

for index_z in z_range:
    logInt_FilePath = root + '/' + string_DataId + '_3d_view.tif'
    memmap_logInt = tifffile.memmap(logInt_FilePath, mode='r')  # Dimension (Y, Z, X)
    logInt_enface = memmap_logInt[:, index_z, ...]
    croplogInt_enface = logInt_enface * enface_mask_to_vol[index_z, ...]  # margin zeros should not be passed to cellpose, otherwise indexing error will raise
    ax1['b'].clear();  ax1['b'].imshow(croplogInt_enface, cmap='gray')
    try:
        # #todo: temporarily remove cellpose filter, to completely exclude low-intensity pixels.
        # croplogInt_enface_seg = croplogInt_enface[overlapRect[0][1]:overlapRect[1][1], overlapRect[0][0]:overlapRect[2][0]]
        # _, _, _, croplogInt_enface_seg_dn = model.eval(croplogInt_enface_seg, diameter=None, channels=[0, 0]) # , niter=200000)
        # _ = enface_mask_to_vol[index_z, ...].copy()
        # cropIntFrameDn = enface_mask_to_vol[index_z, ...].astype('float')
        # cropIntFrameDn[overlapRect[0][1]:overlapRect[1][1], overlapRect[0][0]:overlapRect[2][0]] = croplogInt_enface_seg_dn[..., 0]
        # # ax1['b'].clear();  ax1['b'].imshow(cropIntFrameDn, cmap='gray')
        # frameMask = convert_arr_grayscale_to_dB(cropIntFrameDn, octRangedB) > int_threshold_dB  # create binary mask using threshold on dBscale log int
        frameMask = convert_arr_grayscale_to_dB(croplogInt_enface, octRangedB) > int_threshold_dB
        """ Display masked RGB image: keep pixels where mask is True, black elsewhere. """
        masked_rgb = logInt_enface.copy()
        if masked_rgb.ndim == 3 and masked_rgb.shape[2] >= 3:
            masked_rgb[~frameMask] = 0
            ax1['c'].clear();  ax1['c'].imshow(masked_rgb)
        else:
            ax1['c'].clear();  ax1['c'].imshow(frameMask, cmap='gray')  # Fallback to show mask in grayscale if input is not RGB

        """ Apply `frameMask` to rawLIV en-face image. """
        rawLivFrame = memmap_first_metric[:, index_z, :]  # if VolFlip:  rawLivFrame = tifffile.imread(rawLivFilePath, key=z).
        rawLivFrame_mask = rawLivFrame * frameMask
        rawLivFrame_mask[rawLivFrame_mask == 0] = np.nan
        # ax1['c'].clear();  ax1['c'].imshow(rawLivFrame_mask, cmap='gray')
        cntLiving = np.nansum(rawLivFrame_mask > viability_liv_threshold)  # print('Living count is: ', str(cntLiving))
        cntDead = np.nansum(rawLivFrame_mask < viability_liv_threshold)  # print('Dead count is: ', str(cntDead))
        cntAllPix = cntDead + cntLiving # np.count_nonzero(~np.isnan(rawLivFrame_mask))  # print('All pixel number is: ', str(cntAllPix))
        viability = cntLiving / cntAllPix  # print('Residual missed count is: ', str(cntAllPix - cntDead - cntLiving))

        # # # # * * * * * * * BULLSHIT func: compute mean logInt (maybe linInt) from the cells  * * * * * * * * #
        # ax1['d'].set_ylim(15, 25); ax1['d'].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(1.0))
        # rawLivFrame_mask = logInt_enface * frameMask
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
        if plot_mode == 'scatter':
            second_metric_frame = memmap_second_metric[:, index_z, :].astype(np.float32)  # grayscale enface log int
            # # #todo: （当second_metric='swiftness'时常见）把拟合的swiftness中的所有infinity强制设为50. 需要看看为啥gpufit得到的1/tau是无限
            second_metric_frame[second_metric_frame == np.inf] = np.nan if exclude_inf_swiftness else second_metric_range[1] - 1
            second_metric_mask = second_metric_frame * frameMask  # frameMask = 0 or 1
            second_metric_mask[second_metric_mask == 0] = np.nan  # masked-out or grayscale==0 pixels are nan
            ax1['c'].clear(); ax1['c'].imshow(second_metric_mask, cmap='inferno')
        elif plot_mode == 'counting':
            ax1['d'].scatter((index_z - z_range[0]) * 2, viability, color='#6ea6db', marker='o', s=7);  manual_pick = False

        """ Scatter plot of all masked pixels (mpl_scatter_density to reduce drawing burden) using (modified) LIV and mean frequency. """
        if (plot_mode == 'scatter') and (manual_pick is False):
            y_first_metric = rawLivFrame_mask.flatten()
            x_second_metric = second_metric_mask.flatten()
            valid_mask = ~np.isnan(y_first_metric) & ~np.isnan(x_second_metric)  # np.isinf(valid_mask).any() > False
            if second_metric_horizontal == 'mean intensity': x_second_metric = convert_arr_grayscale_to_dB(x_second_metric, octRangedB)
            # # # (1) 所有mask点绘制 scatter，使用mpl_scatter_density
            self_cmap = LinearSegmentedColormap.from_list('self_cmap', [(0, '#ffffff'), (0.07, '#ffffff'), (0.071, '#160c57'),
                (0.2, '#2469fd'), (0.4, '#24fdfd'), (0.6, '#94fd24'), (0.8, '#fdce24'), (1, '#fd2f24'), ], N=256)
            density = ax1['d'].scatter_density(x_second_metric[valid_mask], y_first_metric[valid_mask], cmap=self_cmap)
            try:  cb.remove()
            except NameError:  pass
            cb = fig1.colorbar(density, ax=ax1['d'], label='Scatter density')
            ax1['d'].set_ylabel(first_metric_label)
            ax1['d'].set_ylim(first_metric_range)
            ax1['d'].set_xlabel(second_metric_label)
            ax1['d'].set_xlim(second_metric_range)

            """ Use the last en-face for scatter plot with histogram - Figure 2 """
            if index_z == z_range[-1]:
                # if not plt.fignum_exists(2):
                fig2 = plt.figure(2, clear=True, figsize=(5.5, 4))
                ax2 = fig2.subplot_mosaic("111", per_subplot_kw={"1": dict(projection='scatter_density')})
                divider = make_axes_locatable(ax2['1'])
                ax_histx = divider.append_axes("top", size="25%", pad=0.01, sharex=ax2['1'])
                ax_histx.xaxis.set_tick_params(labelbottom=False);  ax_histx.xaxis.set_visible(False);  ax_histx.axis('off')
                ax_histy = divider.append_axes("right", size="25%", pad=0.01, sharey=ax2['1'])
                ax_histy.yaxis.set_tick_params(labelleft=False);  ax_histy.yaxis.set_visible(False);  ax_histy.axis('off')
                """ Make scatter plot of all masked pixels. """
                scatter_color = 'blue'
                y_masked = y_first_metric[valid_mask];   x_masked = x_second_metric[valid_mask]  # already converted to dB
                # norm = ImageNormalize(vmin=first_metric_range[0], vmax=first_metric_range[1], stretch=LogStretch())
                density_dead = ax2['1'].scatter_density(x_masked, y_masked, cmap=self_cmap, dpi=25) #, norm=norm) # color=scatter_color)  # cmap=self_cmap)
                # try:  cb2.remove()
                # except NameError:  pass
                cb2 = fig2.colorbar(density, ax=ax2['1'], label='Scatter density')
                """ Adjust axes. """
                ax2['1'].set_ylabel(first_metric_label)
                ax2['1'].set_ylim(first_metric_range)
                ax2['1'].set_xlabel(second_metric_label)
                ax2['1'].set_xlim(second_metric_range)
                """ Make histogram aside from scatter plot. """
                histx, bins_to_histx = np.histogram(x_masked, bins=30, range=second_metric_range, density=True)
                ax_histx.hist(x_masked, bins=bins_to_histx, density=True, alpha=0.4, color=scatter_color)
                histy, bins_to_histy = np.histogram(y_masked, bins=30, range=first_metric_range, density=True)
                ax_histy.hist(y_masked, bins=bins_to_histy, density=True, alpha=0.4, color=scatter_color, orientation='horizontal')


        # #todo: Manual pick for scatter plot. 2026/02/26: Can be removed?
        elif manual_pick:
            print(f"Frame {index_z}: Manual labeling. Left-click to mark 'livng' as green; right-click to mark 'dead' as red. 'Enter' to complete.")
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
    # j = (index_z - z_range[0] + 1) / len(z_range)
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
plt.show(block=True)