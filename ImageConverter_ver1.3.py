import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tifffile
from configparser import ConfigParser
import cv2
from scipy.ndimage import zoom
from tqdm.auto import tqdm
import os
from tkinter import *
from tkinter import filedialog
matplotlib.use("Qt5Agg")


sys_ivs800 = True  # Set as True if taken by IVS-800
gpu_proc = True

save_view = False  # Set as True if save dB-OCT img as 3D stack file for view
save_video = False  # (Only for dtype='timelapse') set as True if save Int_view img as .mp4
display_proc = False  # Set as True if monitor img during converting

save_tif = True  # Set as True if save intensity img as 3D stack .tiff file in the current folder
if gpu_proc:
    import cupy as np; from cupyx.scipy.ndimage import zoom # import nvTIFF
octRangedB = [-10, 70]  # set dynamic range of log OCT signal display
if sys_ivs800:  octRangedB = [-25, 20]
# root = r"F:\Data_2024\20240626_jurkat\lv-0hr"  # Folder path which containing the raw Data
# DataId = "Storage_20240626_12h35m58s.dat"
# DataFold = root + '\\' + DataId  # Raw data file name, usually it is Data.bin
tk = Tk(); tk.withdraw(); DataFold = filedialog.askopenfilename(filetypes=[("", "*")])  # use TkInterface to catch dir path of data file
DataId = os.path.basename(DataFold);   root = os.path.dirname(DataFold)


dataType = None
[dim_y, dim_z, dim_x,FrameRate] = [0, 0, 0, 30]
aspect_ratio = 1
if DataId[-4:] == '.dat':
    dataType = 'timelapse'
elif DataId[-4:] == '.bin':
    dataType = '3d'
else:
    raise ValueError('Unrecognizable data type')
if dataType is not None:
    if dataType == 'timelapse':  # # # # # # # # # #
        with open(DataFold, mode='rb') as file:  # Read info from header
            header = np.fromfile(file, dtype='>d', count=7*2, offset=0)  # read para from header (128*2 bytes)
        [FrameRate, FileSize, XpixSize, ZpixSize, XScanRange, ZScanRange, RI] = header[0:7].tolist()
        [dim_y, dim_x, dim_z] = [int(x) for x in header[1:4].tolist()]  # initialize dim_y, x, z
    elif dataType == '3d':  # # # # # # # # # #
        conf = ConfigParser()
        conf.read(root + '\\' + "DataInfo.ini")
        [dim_x, dim_y, dim_z] = [int(conf['Dimensions']['X']), int(conf['Dimensions']['Y']),
                                 int(conf['Dimensions']['Z'])]  # Dimensions imported from DataInfo.ini
        [XScanRange, ZScanRange] = [float(conf['VoxelSize']['X'])/1000, float(conf['VoxelSize']['Z'])/1000]
        with open(DataFold, mode='rb') as file:
            rawDat = np.fromfile(file, dtype='>f')  # Read data handler. ">": big-endian; "f", float32, 4 bytes
            vol = rawDat.reshape(dim_y, dim_x, dim_z)  # Reshape data as 3D volume
    [pix_x, pix_z] = XScanRange/dim_x*1000, ZScanRange/dim_z*1000  # pixel size in um
    aspect_ratio = pix_x/pix_z
    res_dim_x = int(dim_x*aspect_ratio+0.5)

if save_view or save_tif:
    octImgVol = np.zeros([dim_y, dim_z, res_dim_x]).astype(dtype='f4')  # (dim_y, dim_x, dim_z)
if save_video and dataType == 'timelapse':
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(root + '\\' + DataId[:-4] + '_' + dataType + '_video.avi', fourcc, round(FrameRate), (res_dim_x, dim_z))

# dim_y = 1
for index in tqdm(range(dim_y)):
    if dataType == 'timelapse':
        if sys_ivs800:
            with open(DataFold, mode='rb') as file:  # IVS-800: dtype='>f4'
                data = np.fromfile(file, dtype='>f4', count=dim_x * dim_z, offset=(index * dim_x * dim_z + 64) * 4)
            if data.size < (dim_x * dim_z):
                break
            octImg = data.reshape(dim_x, dim_z).T
        else:
            with open(DataFold, mode='rb') as file:  # IVS-2000: dtype='>i2'
                data = np.fromfile(file, dtype='>i2', count=dim_x * dim_z, offset=(index * dim_x * dim_z + 128) * 2)
            if data.size < (dim_x * dim_z):
                break
            octImg = data.reshape(dim_x, dim_z).T.astype(dtype='f4') / 256 + 80  # convert to scale (only for IVS-2000
    elif dataType == '3d':
        octImg = vol[index, :, :].T

    res_octImg = zoom(octImg, [1, aspect_ratio], order=1)
    if save_view or save_tif:
        octImgVol[index, :, :] = res_octImg
    if save_video and dataType == 'timelapse':
        octImgVideo = (np.clip((res_octImg - octRangedB[0]) / (octRangedB[1] - octRangedB[0]), 0, 1)
                       * 255).astype(dtype='uint8')
        gray_3c = cv2.merge([octImgVideo, octImgVideo, octImgVideo])
        out.write(gray_3c)
        # cv2.imshow('Video', gray_3c)
        # c = cv2.waitKey(1)
        # if c == 27:  break
    if display_proc:
        plt.figure(1, figsize=(dim_x/dim_z*aspect_ratio*5, 5)); plt.clf();
        if dataType == 'timelapse':
            plt.text(0, -10, 'frame = ' + str(index+1) + ' / ' + str(dim_y) + ' ,  '
                     + 'time = ' + str(round(1/FrameRate*index, 2)) + ' s')
        if gpu_proc: res_octImg_plt = res_octImg.get().astype(float)
        else: res_octImg_plt = res_octImg
        plt.imshow(res_octImg_plt, cmap='gray', vmin=octRangedB[0], vmax=octRangedB[1])
        plt.pause(0.01)
        # plt.figure(2); plt.clf();  plt.plot(octImg[10, :])

if save_view:
    if dataType == '3d': res_octImgVol = zoom(octImgVol, [aspect_ratio, 1, 1], order=1)
    elif dataType == 'timelapse': res_octImgVol = octImgVol
    octImgView = (np.clip((res_octImgVol - octRangedB[0]) / (octRangedB[1] - octRangedB[0]),0, 1) * 255).astype(dtype='uint8')
    tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + dataType + '_view.tif', np.rollaxis(octImgView[:, :, :], 0,1).get())
if save_tif:
    res_octImgVol = np.power(10, octImgVol/10)  # convert to linear intensity (square of signal amplitude)
    # if gpu_proc:
    tifffile.imwrite(root + '\\' + DataId[:-4] + '_IntImg.tif', np.rollaxis(res_octImgVol[:, :, :], 0,1).get().astype(dtype='float32'))#, compression='zlib', compressionargs={'level': 8})
if save_video and dataType == 'timelapse':
    out.release();     cv2.destroyAllWindows()

#######
