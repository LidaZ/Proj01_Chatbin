import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tifffile
from configparser import ConfigParser
import cv2
import time
import sys
# from scipy.ndimage import zoom
# from tqdm import tqdm
import os
import gc
import array
import mmap
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
matplotlib.use("Qt5Agg")


"""
Open source project for cell counting system. 
Convert raw data (note: linear OCT intensity, not complex OCT signal) to log and linear intensity 3-D volumes.  
Author: Yijie, Lida

How to use:
Run > Select

Parameters setting: 
rasterRepeat = 1 when: 
    time-lapse data (Storage_xxxx/xx/xx_xxhxxmxxs.dat)
    3-D volume data (Data.bin, taken when rasterRepeat=1)   
rasterRepeat = 32 when: 
    Raster scan data (Data.bin, taken when rasterRepeat=32)
    
multiFolderProcess = False when:
    process multiple 3-D volume or time-lapse data
    not applicable for multiple raster data (maybe)
"""


def return_datatype(dataid):
    dtype = None
    if dataid[-4:] == '.dat':  dtype = 'timelapse'
    elif dataid[-4:] == '.bin':  dtype = '3d'
    else: dtype = None # raise ValueError('Unrecognizable data type')
    return dtype

def close_window():
    global entry
    entry = E.get()
    patch.quit()

def checkFile(path):
    try: os.remove(path)
    except OSError: pass


sys_ivs800 = True
Raster_Repeat_num = 32
multiFolderProcess = False  # if multiple data folders

save_view = True  # Set as True if save dB-OCT img as 3D stack file for view
save_tif = True  # Set as True if save intensity img as 3D stack .tiff file in the current folder

save_video = False  # (Only for dtype='timelapse') set as True if save Int_view img as .mp4
display_proc = False  # Set as True if monitor img during converting
gpu_proc = True

batch_initial_limit = 2.5  # GB, set the file size limit exceeding which enabling batch process
proc_batch = 1


tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True);
ifRaster = messagebox.askyesno(title=None, message='"Yes" if Raster volume; "No" if standard 2D/3D data', parent=tk)
tk.destroy()
if ifRaster: rasterRepeat = Raster_Repeat_num
else: rasterRepeat = 1


# tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True);
# sys_ivs800 = messagebox.askyesno(title=None, message='"Yes" if IVS-800 raw data; "No" if IVS-2000')
# tk.destroy()


if gpu_proc:
    import cupy as np;  # from cupyx.scipy.ndimage import zoom # import nvTIFF (failed)
octRangedB = [-10, 50]  # set dynamic range of log OCT signal display
if sys_ivs800:  octRangedB = [-5, 25]
[dim_y, dim_z, dim_x, FrameRate] = [0, 0, 0, 30];  # aspect_ratio = 1
# # # - - - - fetch dir path of data file - - - - # # #
if multiFolderProcess:
    tk = Tk(); tk.withdraw(); Fold_list = []; DataFold_list = []; extension = ['.dat', '.bin']
    folderPath = filedialog.askdirectory()
    Fold_list.append(folderPath)
    while len(folderPath) > 0:
        folderPath = filedialog.askdirectory(initialdir=os.path.dirname(folderPath))
        if not folderPath:  break
        Fold_list.append(folderPath)
    for item in Fold_list:  # list all files contained in each folder
        fileNameList = os.listdir(item)
        for n in fileNameList:
            if any(x in n for x in extension):
                DataFold_list.append( os.path.join(item, n) )
    FileNum = len(DataFold_list)

else:
    tk = Tk(); tk.withdraw(); DataFold_list = filedialog.askopenfilename(filetypes=[("", "*.bin")], multiple = True)
    tk.destroy()
    FileNum = np.shape(DataFold_list)[0]

for FileId in range(FileNum):
    DataFold = DataFold_list[FileId]
    DataId = os.path.basename(DataFold);   root = os.path.dirname(DataFold)
    dataType = return_datatype(DataId)

    # # # - - - check if Tiff stack file exist - - - # # #
    checkFile(root + '\\' + DataId[:-4] + '_IntImg.tif')
    checkFile(root + '\\' + DataId[:-4] + '_3d_view.tif')
    sys.stdout.write('\n')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(0), 0) + ' initialize processing' + ': '+str(FileId+1)+'/'+str(FileNum))

    # # # - - - initialize data parameters from config/header, enable memory mapping - - - # # #
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
            if rasterRepeat > 1: dim_y_raster = int(dim_y / rasterRepeat)
            [XScanRange, ZScanRange] = [float(conf['VoxelSize']['X'])/1000, float(conf['VoxelSize']['Z'])/1000]
            with open(DataFold, mode='rb') as file:
                # rawDat = np.fromfile(file, dtype='>f')  # Read data handler. ">": big-endian; "f", float32, 4 bytes
                # vol = rawDat.reshape(dim_y, dim_x, dim_z)  # Reshape data as 3D volume
                rawDat = mmap.mmap(file.fileno(), length=0, tagname=None, access=mmap.ACCESS_READ)  #, offset=(any integer)*mmap.ALLOCATIONGRANULARITY)
                # # # use memory mapping to load raw data from large file.
        [pix_x, pix_z] = XScanRange/dim_x*1000, ZScanRange/dim_z*1000  # pixel size in um
        # aspect_ratio = pix_x/pix_z
        # res_dim_x = int(dim_x*aspect_ratio+0.5)

    # # # - - - - enable batch process when engaging large file - - - # # #
    # # # # # - - - - get process batch number - - - # # # # #
    if dataType == 'timelapse':
        proc_batch = 1
    elif dataType == '3d':
        if np.shape(rawDat)[0]/1e9 > batch_initial_limit:  # if file size is larger than 2GB, enable batch process
            proc_batch = 2 ** ( int(np.floor(np.shape(rawDat)[0]/1e9)) - 0 ).bit_length()  # find the smallest power of 2 greater than file size in GB as batch number
            if dim_y % proc_batch != 0:
                raise ValueError('Y dimension is ' + str(dim_y) + ', reset process batch number to make dim_y divisible')
                patch = Tk();  E = Entry(patch);  E.pack();  B = Button(patch, text = "Reset batch number", command = close_window);  B.pack()
                patch.mainloop();   patch.destroy();  proc_batch = int(entry)

    # # # - - - create empty arrays for log-int image storage and plot instant for display - - - # # #
    if save_video and dataType == 'timelapse':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(root + '\\' + DataId[:-4] + '_' + dataType + '_video.avi', fourcc, round(FrameRate), (dim_x, dim_z))
    if display_proc:
        plt.figure(1, figsize=(dim_x/dim_z*7, 7))  # (dim_x/dim_z*aspect_ratio*7, 7)
        plt.gca().set_axis_off(); plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_view or save_tif:
        if rasterRepeat > 1:  octImgVol_raster = np.zeros([int(dim_y/rasterRepeat), dim_z, dim_x]).astype(dtype='f4')
        # # # - - get data size of each batch - - # # #
        dim_y_batch = int(dim_y / proc_batch)
        batch_size = dim_y_batch * dim_x * dim_z * 4  # 4 bytes per element
        octImgVol = np.zeros([dim_y_batch, dim_z, dim_x]).astype(dtype='f4')  # (dim_y, dim_x, dim_z)



    # # # - - - - - - - - - - batch processing - - - - - - - - - # # #
    # proc_batch = 1
    for batch_id in range(proc_batch):
        if dataType == '3d':
            rawDat_batch = rawDat[batch_id*batch_size : (batch_id+1)*batch_size]  # len(rawDat_batch)=batch_size
            # vol = [x[0] for x in struct.iter_unpack('>f', rawDat_batch)]  # iterate too fxxk slow
            vol_tmp = array.array('f', rawDat_batch);   vol_tmp.byteswap()  # convert byte list to '>f' float 1-D array
            vol = np.reshape( np.asarray(vol_tmp), (dim_y_batch, dim_x, dim_z) ) #  reshape 1-D float array to 2-D. Faster than .tolist()
            del vol_tmp  #
    # # # # # - - - - each batch process a sub volume - - - - # # # # #
        # dim_y_batch = 1
        # for index in tqdm(range(dim_y_batch), desc='Batch process is on: '+str(batch_id+1)+'/'+str(proc_batch),
        #                   leave=False, colour='green'):  # got redrawing-bar error when using nested loop for tqdm
        for index in range(dim_y_batch):
            if dataType == 'timelapse':
                if sys_ivs800:
                    with open(DataFold, mode='rb') as file:  # IVS-800: dtype='>f4'
                        data = np.fromfile(file, dtype='>f4', count=dim_x * dim_z, offset=((batch_id * dim_y_batch + index) * dim_x * dim_z + 64) * 4)
                    if data.size < (dim_x * dim_z):
                        break
                    octImg = data.reshape(dim_x, dim_z).T
                else:
                    with open(DataFold, mode='rb') as file:  # IVS-2000: dtype='>i2'
                        data = np.fromfile(file, dtype='>i2', count=dim_x * dim_z, offset=((batch_id * dim_y_batch + index) * dim_x * dim_z + 128) * 2)
                    if data.size < (dim_x * dim_z):
                        break
                    octImg = data.reshape(dim_x, dim_z).T.astype(dtype='f4') / 256 + 80  # convert to scale (only for IVS-2000
            elif dataType == '3d':
                octImg = vol[index, :, :].T
            res_octImg = octImg  # zoom(octImg, [1, aspect_ratio], order=1)
            # # # *extract the 1st image of each repeat raster and save into raster_LogIntView tiffFile # # #
            if save_view or save_tif:
                if rasterRepeat > 1 and (batch_id*dim_y_batch+index)%rasterRepeat == 0:
                    rasterProjInd = int((batch_id*dim_y_batch+index)/rasterRepeat)
                    # octImgVol_raster[rasterProjInd, :, :] = res_octImg  # use first frame of each raster peroid as the log intensity image
                    octImgVol_raster[rasterProjInd, :, :] \
                        = np.max(vol[index:(index+Raster_Repeat_num), :, :].T, 2)  # use max projection of each raster period for log intensity image
            # # # - - - get linear intensity signal into cupy array - - - # # #
            octImgVol[index, :, :] = octImg
            # # # saving octImg instead of res_octImg, to avoid X-interpolation on
            # # # LinearIntImg if zoom enabled for adjusting aspect ratio of B-scan.

            if save_video and dataType == 'timelapse':
                octImgVideo = (np.clip((res_octImg - octRangedB[0]) / (octRangedB[1] - octRangedB[0]), 0, 1)
                               * 255).astype(dtype='uint8')
                gray_3c = cv2.merge([octImgVideo, octImgVideo, octImgVideo])
                out.write(gray_3c)
                # cv2.imshow('Video', gray_3c)
                # c = cv2.waitKey(1)
                # if c == 27:  break
            if display_proc:
                plt.figure(1); plt.clf();
                if dataType == 'timelapse':
                    plt.text(0, -10, 'frame = ' + str((batch_id*dim_y_batch+index)+1) + ' / ' + str(dim_y) + ' ,  '
                             + 'time = ' + str(round(1/FrameRate*(batch_id*dim_y_batch+index), 2)) + ' s')
                if gpu_proc: res_octImg_plt = res_octImg.get().astype(float)
                else: res_octImg_plt = res_octImg
                plt.imshow(res_octImg_plt, cmap='gray', vmin=octRangedB[0], vmax=octRangedB[1]); plt.pause(0.01)

            # # # - - - print progress bar - - - # # #
            total_ind = index + batch_id * dim_y_batch
            sys.stdout.write('\r')
            j = (total_ind + 1) / dim_y
            sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j) + ' on batch processing' + ': '+str(FileId+1)+'/'+str(FileNum))
            sys.stdout.flush()
            time.sleep(0.01)
        # # # - - - check if vol exists when datatype is '3D' - - - # # #
        try:  vol
        except NameError:  vol = None
        else:  del vol
        # # # - - - save LinearIntImg as Tiff stack - - - # # #
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j) + ' saving stack images' + ': '+str(FileId+1)+'/'+str(FileNum))
        time.sleep(0.01)
        octImgVol_rol = np.rollaxis(octImgVol[:, :, :], 0, 1)
        if save_tif:
            octImgVol_linear = np.power(10, octImgVol_rol / 10)  # convert to linear intensity (square of signal amplitude)
            if gpu_proc: octImgVol_sav = octImgVol_linear.get()
            else: octImgVol_sav = octImgVol_linear
            tifffile.imwrite(root + '\\' + DataId[:-4] + '_IntImg.tif', octImgVol_sav.astype(dtype='float32'),
                             append=True, metadata=None) #, bigtiff=True)  # , compression='zlib', compressionargs={'level': 8})

        # # # - - - save LogIntImg as Tiff stack - - - # # #
        if save_view and rasterRepeat == 1:
            octImgView = (np.clip((octImgVol_rol - octRangedB[0]) / (octRangedB[1] - octRangedB[0]), 0, 1) * 255).astype(dtype='uint8')
            if gpu_proc: octImgView_sav = octImgView.get()
            else: octImgView_sav = octImgView
            tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + dataType + '_view.tif', octImgView_sav, append=True)


    # # # - - - - - - for raster scan, compile the first image of each repeat as the 3d_view LogIntImg - - - # # #
    if save_view and rasterRepeat > 1:
        octImgVol_raster_roll = np.rollaxis(octImgVol_raster[:, :, :], 0, 1)
        octImgView_raster = (np.clip((octImgVol_raster_roll - octRangedB[0]) / (octRangedB[1] - octRangedB[0]),0, 1) * 255).astype(dtype='uint8')
        if gpu_proc: octImgView_sav = octImgView_raster.get()
        else: octImgView_sav = octImgView_raster
        tifffile.imwrite(root + '\\' + DataId[:-4] + '_' + dataType + '_view.tif', octImgView_sav)

    if save_video and dataType == 'timelapse':
        out.release();     cv2.destroyAllWindows()


if 'rawDat' in globals(): del rawDat
del octImgVol_rol
del octImgVol
file.close()
gc.collect()
if gpu_proc: np._default_memory_pool.free_all_blocks()
