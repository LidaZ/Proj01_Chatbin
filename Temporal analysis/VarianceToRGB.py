import os
import sys
import numpy as np
from numpy import fft
import gc
import matplotlib.cm as cm
from tqdm.auto import tqdm
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
import matplotlib
matplotlib.use("Qt5Agg")

# # # - - - [1,33多一帧], [34, 65], [66, 97], [98, 129]..., [3938, 3969], [3970, 4000少一帧]- - - # # #
rasterRepeat = 50
# folderPath = r"F:\Data_2024\20240626_jurkat\lv-1hr"  # Open "_IntImg.tif" file
# stackname = "Storage_20240626_13h15m17s_IntImg"  # Storage_20240626_13h24m16s_IntImg / Storage_20240626_13h15m17s_IntImg
# stackFilePath = folderPath + "\\" + stackname + ".tif"
tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True); stackFilePath = filedialog.askopenfilename(filetypes=[("", "*")])
DataId = os.path.basename(stackFilePath);   root = os.path.dirname(stackFilePath);  tk.destroy()
print('Load data folder: ' + root)
fs = 50  # Hz, B-scan frequency during acquisition
cutoff = 0.5;  order = 1  # (cutoff frequency = 0.5, filtering order = 2), lowpass filter to remove DC component before computing variance
colormap = cm.rainbow
norm = matplotlib.colors.Normalize(vmin=0, vmax=2, clip=True)
rawData = tifffile.imread(stackFilePath)   # load linear intensity data from stack. Dimension (Y, Z, X)
dim_y, dim_z, dim_x = np.shape(rawData);  yPosition = dim_y / rasterRepeat
rawDataRotat = np.swapaxes(rawData, 0, 2)  # rotate en-face plane for 90 degree, so that fast scan (X) is horizontal axis to easy 2D FFT. Dimension (X, Z, Y)
varProj = np.zeros((dim_z, dim_x, 3), 'uint8')
varProj_sat = np.ones((dim_x), 'float32')
hueRange = [0, 0.05]  # variance: 0~0.1 / std: 0~
satRange = [0, 0.3]  # intensity: 0~1
 

for depthIndex in tqdm(range(dim_z)): # dim_z
    for yIndex in range(yPosition):
        rawDataRotat_rasterSeg = rawDataRotat[:, :, 0:rasterRepeat]
    # depthIndex = 590  # # #
    enfaceData = rawDataRotat[:, depthIndex, :]  # # (dim_x, dim_y)
    enfaceData_lpfilt = butter_lowpass_filter(enfaceData, cutoff, fs, order)
    enfaceData_lp = enfaceData - enfaceData_lpfilt
    # plt.figure(11);  plt.clf();  plt.plot(enfaceData[548, :]); plt.axis([0,350,-1,6]); # plt.plot(enfaceData_lp[550, :])

    varProj_proj = np.max(enfaceData, axis=1)  # max projection along Y (time)
    varProj_val = np.clip((varProj_proj - satRange[0]) / (satRange[1] - satRange[0]), 0, 1)
    varProj_var = np.var(enfaceData_lp, axis=1)  # np.var()  /  np.std()
    varProj_var_norm = np.divide(varProj_var, np.square(varProj_proj))  # variance normalized by max intensity *SQUARE*, should be robust to noise
    varProj_hue = np.multiply(np.clip((varProj_var_norm - hueRange[0]) / (hueRange[1] - hueRange[0]), 0, 1), 0.6)  # limit color display range from red to blue
    varProj_rgb = hsv_to_rgb(np.transpose([varProj_hue, varProj_val, varProj_val]))  # [varProj_hue, varProj_sat, varProj_val]
    varProj[depthIndex, :, :] = varProj_rgb * 255

    # varProj_rgb = num_to_rgb(np.var(enfaceData_lp, axis=1), max_val=20)
    # varProj_hsv = colorsys.rgb_to_hsv(varProj_rgb[0], varProj_rgb[1], varProj_rgb[2])
    # varProj[depthIndex, :, ] = np.transpose(varProj_rgb)
    # print(str(depthIndex) + '/' + str(dim_z))

figsize_mag = 7
plt.figure(16, figsize=(figsize_mag, dim_z/dim_x*figsize_mag));  plt.clf();  plt.imshow(varProj, vmin=0, vmax=1)
plt.gca().set_axis_off(); plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
gc.collect()


# from scipy.stats import linregress
import numpy as np
from sklearn.metrics import r2_score as linregress
def rsquare(measure, predict):
    e1 = np.subtract(measure, predict)**2
    e2 = np.subtract(measure, np.mean(measure))**2
    r2 = 1 - np.sum(e1) / np.sum(e2)
    return r2

# nc = [13.5, 12.7, 12.6, 12.3, 13.2, 12.5, 12.1, 12]
# oct = [12.5, 11.2, 11.9, 13.2, 12.55, 11.3, 11.5, 12.5]  # [12.5, 12.5, 12.65, 13.2, 13.05, 12.3, 12, 12.5]
# plt.figure(21); plt.clf(); plt.scatter(oct, nc);  plt.axis([11, 14, 11, 14])
# r2 = rsquare(oct, nc)
# print('R2 is: ', r2)
