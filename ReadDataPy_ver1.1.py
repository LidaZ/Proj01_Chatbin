import numpy as np
import matplotlib.pyplot as plt
import tifffile
from configparser import ConfigParser
import time
import cv2
from tqdm.auto import tqdm


save_view = False  # Set as True if save dB-OCT img as 3D stack file for view
save_tif = False  # Set as True if save intensity img as 3D stack .tiff file in the current folder
save_video = False  # Set as True if save Int_view img as .mp4
display_proc = True  # Set as True if monitor img during converting
root = r"C:\Users\lzhu\Downloads\Sample1(Jurkit)"  # Folder path which containing the raw Data
DataId = "Sample1_15_30sec.dat"

octRangedB = [-10, 70]  # set dynamic range of log OCT signal display
DataFold = root + '\\' + DataId  # Raw data file name, usually it is Data.bin
with open(DataFold, mode='rb') as file:  # Read info from header
    header = np.fromfile(file, dtype='>d', count=7*2, offset=0)  # read para from header (128*2 bytes)
[FrameRate, FileSize, XpixSize, ZpixSize, XScanRange, ZScanRange, RI] = header[0:7].tolist()
[dim_y, dim_x, dim_z] = [int(x) for x in header[1:4].tolist()]  # initialize dim_y, x, z

if save_view or save_tif:
    octImgVol = np.zeros([dim_y, dim_z, dim_x]).astype(dtype='f4')  # (dim_y, dim_x, dim_z)
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(root + '\\' + DataId[:-4] + '_timelapse_video.avi', fourcc, 30, (dim_x, dim_z))

for index in tqdm(range(dim_y)):
    with open(DataFold, mode='rb') as file:
        data = np.fromfile(file, dtype='>i2', count=dim_x * dim_z,
                           offset=(index * dim_x * dim_z + 128) * 2)
    if data.size < (dim_x * dim_z):
        break
    octImg = data.reshape(dim_x, dim_z).T.astype(dtype='f4') / 256 + 80  # convert
    if save_view or save_tif:
        octImgVol[index, :, :] = octImg
    if save_video:
        octImgVideo = (np.clip((octImg - octRangedB[0]) / (octRangedB[1] - octRangedB[0]), 0, 1)
                       * 255).astype(dtype='uint8')
        gray_3c = cv2.merge([octImgVideo, octImgVideo, octImgVideo])
        out.write(gray_3c)
        # cv2.imshow('Video', gray_3c)
        # c = cv2.waitKey(1)
        # if c == 27:
        #     break
    if display_proc:
        plt.figure(1); plt.clf();
        plt.text(0, -10, 'frame = ' + str(index+1) + ' / ' + str(dim_y) + ' ,  '
                 + 'time = ' + str(round(1/FrameRate*index, 2)) + ' s')
        plt.imshow(octImg, cmap='gray', vmin=-10, vmax=70)
        plt.pause(0.01)


if save_view:
    octImgView = (np.clip((octImgVol - octRangedB[0]) / (octRangedB[1] - octRangedB[0]),0, 1) * 255).astype(dtype='uint8')
    tifffile.imwrite(root + '\\' + DataId[:-4] + '_timelapse_view.tif', np.rollaxis(octImgView[:, :, :], 0,1))
if save_tif:
    tifffile.imwrite(root + '\\' + DataId[:-4] + '_timelapse.tif', np.rollaxis(octImgVol[:, :, :], 0,1))
if save_video:
    out.release();     cv2.destroyAllWindows()