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

dataFileId = r"E:\20220608\055_abs2"
#dataFileId = r"G:\20220419_hyracotherium\tanishi\tanishi002_abs2_test"

#-------------------------------
# Save color images
#-------------------------------

newLoad = 1
if newLoad == 1:
    LIV = np.load(dataFileId+'_LIV.npy')
    INT = np.load(dataFileId+'_LogIntensity.npy')




# LIV = LIV[:,:,250:800]
#INT = INT[:,:,500:1024]

# Input and output image ranges as [INT, LIV]    
inRanges = [(50., 80.), (0., 30.)]
outRanges = [(0., 1.), (0., 120.)]


rgbImage = np.zeros((LIV.shape[0], LIV.shape[1], LIV.shape[2],3))
intImage = np.zeros((LIV.shape[0], LIV.shape[1], LIV.shape[2],3))
for bscanId in range(0,LIV.shape[0]):
    
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


    
tifffile.imsave(dataFileId+'_LivColor_vol_kernel3_LIV'+str(inRanges[1])+'_int'+str(inRanges[0])+'.tiff', rgbImage.astype('uint8'), photometric='rgb',compress=6)
tifffile.imsave(dataFileId+'_Int_vol_int'+str(inRanges[0])+'.tiff', intImage.astype('uint8'), photometric='rgb',compress=6)
