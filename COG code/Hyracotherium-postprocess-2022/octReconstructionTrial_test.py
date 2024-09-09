# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:29:18 2023

@author: rionm
"""

import numpy as np
import cogCorrelation as corr
import time
from skimage.registration import phase_cross_correlation
import scipy.ndimage as scim
import tifffile
import dOct
import octReconstructionTrial as octRec
import matplotlib.pyplot as plt


myRaster = dOct.rasterParamGeneralized(aPerFrame = 512,
                                           frameRepeats = 1,
                                           bscanPerBlock = 16,
                                           blockRepeats = 32,
                                           blockPerVolume = 8)        
        
data = octRec.octVolume(filePath = r'L:/20230907/002.dat',
                 rasterParamGeneralized = myRaster,
                 fileType = 'dat',
                 systemType = 'Hyracotherium')

spectra2D = data.frameSpectra(bIndex = 1984)

octImage = np.fft.fft(spectra2D,axis = 1)

plt.imshow(np.log(np.abs(octImage)))
plt.plot(spectra2D[100,:])

