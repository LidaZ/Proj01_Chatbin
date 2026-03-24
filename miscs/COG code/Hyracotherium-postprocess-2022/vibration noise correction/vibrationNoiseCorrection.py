# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:50:20 2022

@author: rionm
"""
import time
start_time= time.time()
import os
import numpy as np
import scipy.ndimage as scim
import matplotlib.pyplot as plt

#from skimage import data
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
#from scipy.ndimage import shift

#from skimage.feature import register_translation
# from skimage.feature.register_translation import _upsampled_dft
#from scipy.ndimage import fourier_shift


pixPerA = 1024
aPerB = 512         
pixByte = 4
framePerVolume = 4096
frameSize = pixPerA*aPerB    

#dataPath = r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\030_abs2.3dv"
#dataPath = r"G:\20220614\067_abs2.3dv"
#dataPath = r"G:\20220623(processed data)\047_abs2.3dv"
dataPath = r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\067_abs2.3dv"
root = os.path.splitext(dataPath)[0]
# flg_load = False1
# if flg_load == True:
#     Data = np.zeros((framePerVolume, pixPerA,aPerB), dtype='float32')
#     for frameId in range (0, framePerVolume):
#             tmpData = np.fromfile(file = dataPath, dtype='>f4', 
#                                       count = frameSize, offset = frameSize*frameId*pixByte)
#             tmpData = tmpData.reshape(aPerB, pixPerA)
#             tmpData = np.transpose(tmpData)
#             Data[frameId,:,:] = tmpData
#     np.save(r'D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\Data',Data)
# else:
# #    pass
#     np.load(r'D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\Data.npy')

def getFrameIndexSequence(bIndex, bPerBlock, framePerB):
    # bIndex = 30
    # bPerBlock = 16
    # framePerB = 32
    framePerBlock = bPerBlock*framePerB
    blockIndex = int(bIndex / bPerBlock)
    bIdxInBlock = bIndex - blockIndex * bPerBlock
    firstFrame = blockIndex * framePerBlock + bIdxInBlock
    frameIndexes = range(firstFrame,firstFrame+framePerBlock, bPerBlock)
    return list(frameIndexes)

version = 1 # 0 for select one Bscan, 1 for select all B-scan
validationFlag = 1
if version ==0: # for one B-scan
    # Select B-scan
    bIndex = 30
    bPerBlock = 16
    framePerB = 32
    # framePerBlock = bPerBlock*framePerB
    # blockIndex = int(bIndex / bPerBlock)
    # bIdxInBlock = bIndex - blockIndex * bPerBlock
    # firstFrame = blockIndex * framePerBlock + bIdxInBlock
    
    DataOrig = np.zeros((framePerB, pixPerA,aPerB), dtype='float32')
    idx = 0
    #for frameId in range(firstFrame,firstFrame+framePerBlock, bPerBlock):
    for frameId in getFrameIndexSequence(bIndex, bPerBlock, framePerB):
            tmpData = np.fromfile(file = dataPath, dtype='>f4', 
                                      count = frameSize, offset = frameSize*frameId*pixByte)
            tmpData = tmpData.reshape(aPerB, pixPerA)
            tmpData = np.transpose(tmpData)
            DataOrig[idx,:,:] = tmpData
            idx += 1
    
    DataShifted = np.zeros(DataOrig.shape)
    shiftVal = np.zeros((framePerB,2))
    for fIndex in range(0,framePerB):
    #for fIndex in range(0,2):
        shift, error, diffphase = phase_cross_correlation(DataOrig[0], DataOrig[fIndex], upsample_factor=100)
        shiftVal[fIndex] = shift
        # Option-1: Use log for interpolation
        DataShifted[fIndex] = scim.shift(10*np.log10(DataOrig[fIndex]), shiftVal[fIndex])
        DataShifted[fIndex] = 10**(DataShifted[fIndex]/10)
        # Option-2: Use linear for interpolation
        #DataShifted[fIndex] = scim.shift(DataOrig[fIndex], shiftVal[fIndex])
    vibEnergy = np.var(shiftVal[:,0])+ np.var(shiftVal[:,1])
    
    shiftVal2 = np.zeros((framePerB,2))
    for fIndex in range(0,framePerB):
    #for fIndex in range(0,2):
        shift, error, diffphase = phase_cross_correlation(DataShifted[0], DataShifted[fIndex], upsample_factor=100)
        shiftVal2[fIndex] = shift
    vibEnergy2 = np.var(shiftVal2[:,0])+ np.var(shiftVal2[:,1])
    
    
    print("Vibration Energy original  = ", str(vibEnergy))
    print("Vibration Energy corrected = ", str(vibEnergy2))
    
    # np.save(root + "_shiftCorrected", Data2)
    # np.save(root + "_vibEnergy_original", vibEnergy)
    # np.save(root + "_vibEnergy_corrected", vibEnergy2)
    
    #plt.scatter(list([1,1]),vibEnergyArray)
    #plt.imshow(Data[1])


if version == 1: #for all B-scan
    bPerBlock = 16
    framePerB = 32
    bPerVolume = int(framePerVolume/framePerB)
    vibEnergyArray = np.zeros((2, bPerVolume))
    DataShifted3d = np.zeros((framePerVolume, pixPerA, aPerB))
    for locId in range(0, bPerVolume):
        bIndex = locId
        framePerBlock = bPerBlock*framePerB
        blockIndex = int(bIndex / bPerBlock)
        bIdxInBlock = bIndex - blockIndex * bPerBlock
        firstFrame = blockIndex * framePerBlock + bIdxInBlock
        DataOrig = np.zeros((framePerB, pixPerA,aPerB), dtype='float32')
        idx = 0
        for frameId in range(firstFrame,firstFrame+framePerBlock, bPerBlock):
                tmpData = np.fromfile(file = dataPath, dtype='>f4', 
                                          count = frameSize, offset = frameSize*frameId*pixByte)
                tmpData = tmpData.reshape(aPerB, pixPerA)
                tmpData = np.transpose(tmpData)
                DataOrig[idx,:,:] = tmpData
                idx += 1
        
        DataShifted = np.zeros(DataOrig.shape)
        shiftVal = np.zeros((framePerB,2))
        for fIndex in range(0,framePerB):
            shift, error, diffphase = phase_cross_correlation(DataOrig[15], DataOrig[fIndex], upsample_factor=10)
            shiftVal[fIndex] = shift
            # Option-1: Use log for interpolation
            DataShifted[fIndex] = scim.shift(10.0*np.log10(DataOrig[fIndex]), shiftVal[fIndex])
            DataShifted[fIndex] = 10.0**(DataShifted[fIndex]/10.0)
            # Option-2: Use linear for interpolation
            #DataShifted[fIndex] = scim.shift(DataOrig[fIndex], shiftVal[fIndex])
            DataShifted3d[list(range(firstFrame,firstFrame+framePerBlock, bPerBlock))[fIndex],:,:] = DataShifted[fIndex]
        if validationFlag == 1:    
            vibEnergy = np.var(shiftVal[:,0])+ np.var(shiftVal[:,1])
            vibEnergyArray[0, locId] = vibEnergy    
            shiftVal2 = np.zeros((framePerB,2))
            for fIndex in range(0,framePerB):
                shift, error, diffphase = phase_cross_correlation(DataShifted[15], DataShifted[fIndex], upsample_factor=10)
                shiftVal2[fIndex] = shift
            vibEnergy2 = np.var(shiftVal2[:,0])+ np.var(shiftVal2[:,1])
            vibEnergyArray[1, locId] = vibEnergy2
        print ("location = " + str(locId))
if validationFlag == 1:        
    vibEnergyArrayT = vibEnergyArray.transpose()
    plt.scatter(list(range(1,129)), vibEnergyArrayT[:,0], label ="original")
    plt.scatter(list(range(1,129)), vibEnergyArrayT[:,1], label = "corrected")
    #plt.ylim(0,0.25)
    plt.legend(loc='upper left',prop={'size':10})
    plt.xlabel('Frame location in volume',size='10')
    plt.ylabel('VibrationEnergy',size='10')

DataShifted3d = np.transpose(DataShifted3d.astype(np.float32), axes = (0,2,1))
#np.save(r"D:/programs/Hyracotherium-postprocess-2022/vibration noise correction/data/20220908/067_abs2_corrected",DataShifted3d)
end_time = time.time()
print("time = " + str(end_time-start_time) + " s")
