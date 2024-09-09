# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 22:01:22 2022

@author: rionm
"""
import cv2
import tifffile
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import imagecolorizer as icz

def savePsudoColorImage(dataFilePath, OCDSdataFilePath, newLoad, inRanges, outRanges):
    if newLoad == 1:
        #LIV-------------------
        LIV = np.load(dataFilePath+'_LIV.npy') 
        #------------------
        #OCDS---------------
        #LIV = np.load(OCDSdataFilePath)
        #LIV = LIV.transpose(0, 2, 1)
        #--------------------
        INT = np.load(dataFilePath+'_LogIntensity.npy')
    
    # LIV = LIV[:,:,250:800]
    #INT = INT[:,:,500:1024]
    
    rgbImage = np.zeros((INT.shape[0], INT.shape[1], INT.shape[2],3))
    intImage = np.zeros((INT.shape[0], INT.shape[1], INT.shape[2],3))
    for bscanId in range(0,INT.shape[0]):
        
        print(bscanId)
        imgIndex = 0
        V = icz.valueRerange(INT[bscanId,:,:], inRanges[imgIndex], outRanges[imgIndex])

        imgIndex = 1
        thisLIV = LIV[bscanId,:,:]
     #   thisLIV = gaussian_filter(thisLIV, sigma=[2,2])
        thisLIV = cv2.blur(thisLIV,(3,3))                                     
    #    H = icz.valueRerange(LIV[bscanId,:,:], inRanges[imgIndex], outRanges[imgIndex])
        H = icz.valueRerange(thisLIV, inRanges[imgIndex], outRanges[imgIndex])
        S = np.ones(LIV[bscanId,:,:].shape)
        rgbImage[bscanId,:,:] = icz.hsvToRgbImage(H,S,V)
        intImage[bscanId,:,:] = icz.hsvToRgbImage(H,S*0.,V)

    
    #LIV    
    tifffile.imsave(dataFilePath+'_LivColor_vol_kernel3_LIV'+str(inRanges[1])+'_int'+str(inRanges[0])+'.tiff', rgbImage.astype('uint8'), photometric='rgb',compress=6)
    #OCDS    
    #tifffile.imsave(dataFilePath+'_OcdsColor_vol_kernel3_OCDSl'+str(inRanges[1])+'_int'+str(inRanges[0])+'.tiff', rgbImage.astype('uint8'), photometric='rgb',compress=6)
    tifffile.imsave(dataFilePath+'_Int_vol_int'+str(inRanges[0])+'.tiff', intImage.astype('uint8'), photometric='rgb',compress=6) 
    return()

OCDSdataFilePath = [
    
# r"G:\20220623(processed data)\006_abs2_decay speed[32]6.55362_4Kernelsum_abs_log1[204.8]_6[1228.8].npy",
# r"G:\20220623(processed data)\010_abs2_decay speed[32]6.55362_4Kernelsum_abs_log1[204.8]_6[1228.8].npy",
# r"G:\20220623(processed data)\030_abs2_decay speed[32]6.55362_4Kernelsum_abs_log1[204.8]_6[1228.8].npy",
# r"G:\20220623(processed data)\034_abs2_decay speed[32]6.55362_4Kernelsum_abs_log1[204.8]_6[1228.8].npy",
# r"G:\20220623(processed data)\042_abs2_decay speed[32]6.55362_4Kernelsum_abs_log1[204.8]_6[1228.8].npy",
# r"G:\20220623(processed data)\048_abs2_decay speed[32]6.55362_4Kernelsum_abs_log1[204.8]_6[1228.8].npy"
r"F:\042_abs2_decay speed[32]6.55362_4Kernelsum_abs_log1[204.8]_6[1228.8]_01.npy"

    ]

dataFilePath = [
#r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\20220831\067-05_abs2_corrected"
#r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\20220905\047_abs2_corrected"
#r"F:\042_abs2"
r"D:\programs\Hyracotherium-postprocess-2022\testData\organoid01\047_abs2"
    ]

newLoad = 1 # if this data is that imported newly...=1
# Input and output image ranges as [INT, LIV]    
#LIV----------------------
inRanges = [(43., 70.), (10., 30.)]
#----------------------------
#OCDSl-----------------------
#inRanges = [(43., 70.), (0., 0.0006)]
#-----------------------------
outRanges = [(0., 1.), (0., 120.)]

for dataId in range (0, len(dataFilePath)):
    savePsudoColorImage(dataFilePath[dataId], OCDSdataFilePath[dataId], newLoad, inRanges, outRanges)
    print ('-----------dataID: '+ str(dataId) + ' was processed.------------')


