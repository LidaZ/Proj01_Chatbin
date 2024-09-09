# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:50:24 2022

@author: rionm
"""

import numpy as np
import os
import tifffile
import skimage.io as io

data_path = r"D:\VLIV-test\organoid_hyracotherium\015_abs2.3dv"
root = os.path.splitext(data_path)[0]
pixByte = 4
pixPerA = 1024  #1024
aPerFrame = 512
framePerVolume = 4096
frameSize = pixPerA*aPerFrame

logData = np.zeros((framePerVolume, aPerFrame, pixPerA), dtype='float32')
abs2Data = np.zeros((framePerVolume, aPerFrame, pixPerA), dtype='float32')

for frameId in range (0, framePerVolume):
#for frameId in range (0, 5):
    print("frame ID " + str(frameId) + " is loading.")
    tmpData = np.fromfile(file = data_path, dtype='>f4', 
                              count = frameSize, offset = frameId *frameSize* pixByte) #offset = frameId * pixByte
    tmpData = tmpData.reshape(aPerFrame, pixPerA)

    #logtmpData = 10*np.log10(tmpData)

    #tmpData = logData[frameId,:,:]
    abs2Data[frameId,:,:] = tmpData
    #logData[frameId,:,:] = 10.*np.log10(tmpData)

# tifffile.imsave(root+ 'OCTIntensity_View' +'.tif', logData.astype(dtype='f4'))
tifffile.imsave(root +'.tif', abs2Data.astype(dtype='f4'))



