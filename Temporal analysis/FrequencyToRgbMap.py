import os
import sys
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import gc
# from scipy import stats
# import pingouin as pg
# import pandas as pd
# from scipy.stats import norm
# from scipy.optimize import curve_fit
import tifffile
from PIL import Image
import matplotlib
matplotlib.use("Qt5Agg")


folderPath = r"F:\Data_2024\20240626_jurkat\mv-2hr"
stackname = "Storage_20240626_14h30m37s_IntImg"
stackFilePath = folderPath + "\\" + stackname + ".tif"
rawData = tifffile.imread(stackFilePath)   # load linear intensity data from stack. Dimension (Y, Z, X)
dim_y, dim_z, dim_x = np.shape(rawData)
rawDataRotat = np.swapaxes(rawData, 0, 2)  # rotate en-face plane for 90 degree, so that fast scan (X) is horizontal axis to easy 2D FFT. Dimension (X, Z, Y)
freqEncodeProj = np.zeros((dim_x, dim_z, 3), 'uint8')
fs = 50  # Hz, B-scan frequency during acquisition
band1 = 0.75 ;  band2 = 1.1 ; band3 = 2.2  # Hz, check figure(12)
band_discr = (np.divide([band1, band2, band3], fs) * dim_y).astype('int')

for depthIndex in range(dim_z):   # dim_z
    # depthIndex = 512
    enfaceData = rawDataRotat[:, depthIndex, :];   # plt.figure(10); plt.clf(); plt.imshow(np.abs(enfaceData))
    enfaceFreqSpec = fft.fft(enfaceData[:, :]*np.tile(np.hanning(dim_y), (dim_x, 1)))  # apply Hann window before fft to mitigate frequency leakage
    ampEnfaceFreqSpec = np.abs(enfaceFreqSpec[:, 1:round(dim_y/2)])  # mapping freq range (1:20) # 1:round(np.shape(enfaceFreqSpec)[1]/2)

    # # # # - - - - - - - - - Plot frequency spectrum to determine the frequency bands - - - - - # # #
    # freqs = np.arange(dim_y)/dim_y*fs;  freq_half = freqs[0:round(int(dim_y/2))]  # np.fft.fftfreq(dim_y)
    # plt.figure(11);    plt.clf();    plt.imshow(ampEnfaceFreqSpec)
    # plt.figure(12); freqProfile = np.abs(enfaceFreqSpec[126, 0:round(int(dim_y/2))]); plt.plot(freq_half, freqProfile/freqProfile[0])
    # plt.xlabel('Frequency (Hz)'); plt.ylabel('Normalized power (a.u.)')

    r = np.sum(ampEnfaceFreqSpec[:, 1:band_discr[0]], axis=1);             r_ch = r / 40  # [1:7] [7:17] [17:28] [28:]
    g = np.sum(ampEnfaceFreqSpec[:, band_discr[0]:band_discr[1]], axis=1); g_ch = g / 20  # 0~0.75hz:10; 0.75~5hz:30; 5~25hz:40
    b = np.sum(ampEnfaceFreqSpec[:, band_discr[1]:band_discr[2]], axis=1); b_ch = b / 10
    freqEncodeProj[:, depthIndex, 0] = r_ch*256
    freqEncodeProj[:, depthIndex, 1] = g_ch*256
    freqEncodeProj[:, depthIndex, 2] = b_ch*256

    print(str(depthIndex) + '/' + str(dim_z))

print('r,g,b maxs are: '+str(r.max())+'; '+str(g.max())+'; '+str(b.max()))
img = np.swapaxes(Image.fromarray(freqEncodeProj), 0, 1)
figsize_mag = 7
plt.figure(16, figsize=(figsize_mag, dim_z/dim_x*figsize_mag));  plt.clf();  plt.imshow(img)
plt.gca().set_axis_off(); plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
gc.collect()
