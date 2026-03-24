# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 21:04:45 2022

@author: rionm
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
import importlib
from tifffile import imsave
import rasterVolume as rv
importlib.reload(rv)

def computeLIV(dataVolume, bIndexes = (0)):
    myParam = dataVolume.rasterParam
    
    root = os.path.splitext(dataVolume.filePath)[0]
    print(root)
    #----- GeneralizedRasterParam--------------------------------------------------------

    # Process each B-scan
    if bIndexes == (0):
        bIndexes = range(myParam.bscanPerBlock * myParam.blockPerVolume)
    else:
        pass

    if FlagLIV == 1:
        speckle_variance = np.zeros((len(bIndexes), myParam.aPerFrame, pixPerA),dtype='float32')
    if FlagInt == 1:
        logIntensity     = np.zeros((len(bIndexes), myParam.aPerFrame, pixPerA),dtype='float32')
        
#    for bIndex in range(myParam.bscanPerBlock * myParam.blockPerVolume):
    
    iterIndex = 0
    for bIndex in bIndexes:
        logData = dataVolume.frameSequence(bIndex, outputSignalScale = 'dB')
    

    # if bIndexes == (0):
    #     bIndex = range(myParam.bscanPerBlock * myParam.blockPerVolume)
    # else:
    #     bIndex = bIndexes
        
    # for bIndex in bIndexes:
    #     logData = dataVolume.frameSequence(bIndex)
    
                
    #----------------------------------------------------------------------
        
        M = np.average(logData,axis=0)
    
        if FlagLIV == 1:
            Sub=np.zeros((logData.shape))
            for i in range(logData.shape[0]):
                Sub[i]=(logData[i]- M) 
            
            SUM=np.sum(Sub**2,axis=0)
            speckle_variance[iterIndex]=np.divide(SUM,logData.shape[0])
    
        if FlagInt == 1:
            logIntensity[iterIndex] = M
        
        iterIndex += 1
        print(str(bIndex)+ ' frame location was processed.')
        
        
    if FlagLIV == 1:
        np.save(root+'_LIV.npy', speckle_variance)
        tifffile.imsave(root+ '_LIV.tif', speckle_variance.astype(dtype='f4'))
    if FlagInt == 1:
        np.save(root+'_LogIntensity.npy', logIntensity)
        tifffile.imsave(root+ '_LogIntensity.tif', logIntensity.astype(dtype='f4'))
   
    return()
  
def computeLIV_obsolete(rasterParam, pixByte, pixPerA, filename, FlagLIV, FlagInt, fileType = '3dv'):
    root = os.path.splitext(filename)[0]
    #----- GeneralizedRasterParam--------------------------------------------------------
    if FlagLIV == 1:
        speckle_variance = np.zeros((rasterParam.bscanPerBlock * rasterParam.blockPerVolume, rasterParam.aPerFrame, pixPerA),dtype='float32')
    if FlagInt == 1:
        logIntensity     = np.zeros((rasterParam.bscanPerBlock * rasterParam.blockPerVolume, rasterParam.aPerFrame, pixPerA),dtype='float32')

    # Process each B-scan
    dataVolume = rv.rasterVolume(filePath = filename, rasterParamGeneralized = rasterParam, pixPerA = pixPerA, pixByte = pixByte, fileType = fileType)
    for bIndex in range(rasterParam.bscanPerBlock * rasterParam.blockPerVolume):
        logData = dataVolume.frameSequence(bIndex)
    
            
    #----------------------------------------------------------------------
        
        M = np.average(logData,axis=0)
    
        if FlagLIV == 1:
            Sub=np.zeros((logData.shape))
            for i in range(logData.shape[0]):
                Sub[i]=(logData[i]- M) 
            
            SUM=np.sum(Sub**2,axis=0)
            speckle_variance[bIndex]=np.divide(SUM,logData.shape[0])
    
        if FlagInt == 1:
            logIntensity[bIndex] = M
            
        print(str(bIndex)+ ' frame location was processed.')
        
    if FlagLIV == 1:
        np.save(root+'_LIV.npy', speckle_variance)
        tifffile.imsave(root+ '_LIV.tif', speckle_variance.astype(dtype='f4'))
    if FlagInt == 1:
        np.save(root+'_LogIntensity.npy', logIntensity)
        tifffile.imsave(root+ '_LogIntensity.tif', logIntensity.astype(dtype='f4'))
   
    return()


#---------default(use this when usual processing)--------------------------------------------------------

inputFilePath = [
# r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\20220831\067_abs2_corrected.npy"
 #r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\20220831\067-05_abs2_corrected.npy"
# r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\030_abs2.3dv"
#r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\20220906\047_abs2_corrected.npy"
r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\20220908\047_abs2_corrected.npy"
    ]
fileType = 'npy'
signalScale = 'linear'

flag_dynamics = 1 #scan protocol (dynamics or burst)

#
# Set generalized raster parametrers
#
if flag_dynamics == 1:
    rastParam = rv.rasterParamGeneralized(
        aPerFrame = 512, frameRepeats =1,
        bscanPerBlock = 16, blockRepeats = 32, blockPerVolume = 8) #original(for all location)
        # aPerFrame = 512, frameRepeats =1,
        # bscanPerBlock = 1, blockRepeats = 32, blockPerVolume = 1)  #for one location
else:
    rastParam = rv.rasterParamGeneralized(
        aPerFrame = 512, frameRepeats =1,
        bscanPerBlock = 512, blockRepeats = 1, blockPerVolume = 1)
    
#pixByte = 16 # for complex126
pixByte = 4 # for float32
pixPerA = 1024


#---------Frag--------------------------------------------------
FlagLIV = 1 #compute and save
FlagInt = 1 #compute and save
#--------------------------------------------------------------
for dataId in range (0, len(inputFilePath)):
    filename = inputFilePath[dataId]
    dataVolume = rv.rasterVolume(filePath = filename,
                                 rasterParamGeneralized = rastParam,
                                 pixPerA = pixPerA, pixByte = pixByte,
                                 fileType = fileType,
                                 signalScale = signalScale)
    computeLIV(dataVolume, bIndexes = (0))
    # LivImage = computaLIV(dataVolume)
    # IntImage = computeIntensity(dataVolume)    
    # OcdsImage = computeOcds(dataVolume)
    # savePseudoColorImage(filepath, LivImage, IntImage, cmap)
    # savePseudoColorImage(filepath, OcdsImage, IntImage, cmap)
    print('--------------dataID: '+str(dataId)+ ' was processed.------------------')
"""
fp = DoctFilePath(r"D:/..../inputDataFolder or data") # constructor.
fp.setOutputFolder(r"...", appnedToRoot = True or False)
fp.getFilename(tag, extension) # e.g. fp.getFilename("_LIV", "npy")

savePseudoColivIMage(fp.getFilename("_LIV", "npy"), LivImage, IntImage, cmap)
"""

#----------for low resolution test of LIV (_testForLowRes.vi)---------------------------------------------
# inputFilePath = r"D:\programs\dataForLowResTest\20220722"
# dataName = "064"
# resolutionVariation = 10

# flag_dynamics = 1 #scan protocol (dynamics or burst)

# if flag_dynamics == 1:
#     a = rasterParamGeneralized(
#         # aPerFrame = 512, frameRepeats =1,
#         # bscanPerBlock = 16, blockRepeats = 32, blockPerVolume = 8) #original(for all location)
#         aPerFrame = 512, frameRepeats =1,
#         bscanPerBlock = 1, blockRepeats = 32, blockPerVolume = 1)  #for one location
# else:
#     a = rasterParamGeneralized(
#         aPerFrame = 512, frameRepeats =1,
#         bscanPerBlock = 512, blockRepeats = 1, blockPerVolume = 1)
    
# #pixByte = 16 # for complex126
# pixByte = 4 # for float32
# pixPerA = 1024

# #---------Frag--------------------------------------------------
# FlagLIV = 1 #compute and save
# FlagInt = 1 #compute and save
# #--------------------------------------------------------------

# FilterRes_d = [0.695829, 0.376251, 0.273756, 0.218716, 0.183351, 0.158372,
#                 0.139654, 0.12504, 0.113281, 0.103598]
# FilterRes_h = [0.189483, 0.207776, 0.230183, 0.258354, 0.295027, 0.345153, 0.418931,
#                 0.542315, 0.816268, 1.000000]
# # #------when apply filter for both resolution---------------------

# # for dataId1 in range (0, resolutionVariation ):
# #     for dataId2 in range (0, resolutionVariation ):
# #         # resolution1 = 0.1*(dataId1+1)
# #         # resolution2 = 0.1*(dataId2+1)
# #         resolution1 = FilterRes_d[dataId1]
# #         resolution2 = FilterRes_h[dataId2]
# #         filename = inputFilePath  + "\\" +dataName + "_z" + str('{:.06f}'.format(resolution1)) + "x" + str('{:.06f}'.format(resolution2)) +"_abs2.3dv"
# #         a.computeLIV( pixByte, pixPerA, filename, FlagLIV, FlagInt )
# #         print('--------------dataID1: '+str(dataId1)+"    dataID2:" +str(dataId2)+' was processed.------------------')
# # #-------------------------------------------------------------------

# #-------when apply filter for only one resolution-------------------
# # for dataId1 in range (0, resolutionVariation ):
# #     resolution = FilterRes_h[dataId1]
# #     filename = inputFilePath  + "\\" +dataName + "_z" + "non" + "x" + str('{:.06f}'.format(resolution)) +"_abs2.3dv"
# #     a.computeLIV( pixByte, pixPerA, filename, FlagLIV, FlagInt )
# #     print('---dataID1:  '+str(dataId1))
# for dataId2 in range (0, resolutionVariation ):
#     resolution = FilterRes_d[dataId2]
#     filename = inputFilePath  + "\\" +dataName + "_z" + str('{:.06f}'.format(resolution)) + "x" + "non" +"_abs2.3dv"
#     a.computeLIV( pixByte, pixPerA, filename, FlagLIV, FlagInt )
#     print('---dataID2: '+str(dataId2))
# #-------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------