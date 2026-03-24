# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:51:55 2022

@author: rionm
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile



complexFilePath = r"K:\20220613\157_complex.3dv"
intensityNumpyFile = r"K:\20220613\157_abs2_LogIntensity.npy"

# flag = 1...show image to select ROI
# flag = 0...compute variance of real and imaginary
flag = 0

pixByte = 16
pixPerA = 1024

aPerFrame = 512
framePerVolume = 4096
#ROI
upper = 300 #depth direction
bottom = 400 #depth direction
left = 0 #horizontal direction
right = 100 #horizontal direction

if flag == 1 :
    intNp = np.load(intensityNumpyFile)
    plt.imshow(intNp[62])

if flag == 0 :
    frameSize = aPerFrame * pixPerA
    
    # varRe = np.zeros((framePerVolume),dtype='complex128')
    # varIm = np.zeros((framePerVolume),dtype='complex128')
    noise = np.zeros((framePerVolume),dtype='complex128')
    for frameId in range (0, framePerVolume):
        
        tmpData = np.fromfile(file = complexFilePath, dtype='>f8', 
                              count = frameSize*2, offset = frameId * pixByte)# *2 is to account for complex (f8 + f8)
        tmpData = tmpData.reshape([frameSize,2])
        re = tmpData[:,0]
        im = tmpData[:,1]
        re = re.reshape([aPerFrame, pixPerA])
        im = im.reshape([aPerFrame, pixPerA])
        
        re = re[upper:bottom, left:right] #select ROI
        im = im[upper:bottom, left:right]
        
        # varRe[frameId] = np.var(re)
        # varIm[frameId] = np.var(im)
        noise[frameId] = np.var(re) + np.var(im)
    meanNoise = np.mean(noise)
    meanNoise_dB = 10 * np.log10(np.abs(meanNoise))
    print (meanNoise_dB)
        
        




