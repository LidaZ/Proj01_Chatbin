# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 21:04:45 2022

@author: rionm
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from tifffile import imsave

class rasterParamGeneralized:     
    def __init__(self, aPerFrame, frameRepeats, bscanPerBlock, blockRepeats, blockPerVolume):
        self.aPerFrame = aPerFrame
        self.frameRepeats = frameRepeats
        self.framePullback = 0
        self.frameSettleTime = 0
        self.bscanPerBlock = bscanPerBlock
        self.blockRepeats = blockRepeats
        self.blockPerVolume = blockPerVolume
        self.volumePullback = 0
        self.triggerFrameWidth = 0
        self.triggerFrameDelay = 0
        self.triggerLineWidth = 0
        self.triggerLineDelay = 0
        self.otherSubsample = 0

    def frameSize(self, pixPerA):
        """
        returns frameSize in pix
        """
        return(self.aPerFrame * pixPerA)
    
    def getFrameOffsets(self,frameLocation, pixPerA=2048):
        """
        returns the file offset in data-index (not byte)
        """
        bscanIndex = frameLocation % self.bscanPerBlock
        blockIndex = int(frameLocation/self.bscanPerBlock)
        
        singleBlockSize = self.frameRepeats * self.bscanPerBlock
        fullBlockSize = singleBlockSize * self.blockRepeats
        blockRepeatIndex = np.array(range(0,self.blockRepeats),dtype='uint32')
        frameIndex = fullBlockSize * blockIndex + singleBlockSize * blockRepeatIndex + self.frameRepeats * bscanIndex
        theOffsets = frameIndex * self.frameSize(pixPerA = pixPerA)
        return(theOffsets)

# path_input_array = [
# r"G:\20220415\spheroid\spheroid002_complex.3dv"
# ]

filename = r"G:\20220415\spheroid\spheroid002_complex.3dv"

#----- GeneralizedRasterParam

pixByte = 16 # for 
pixPerA = 2048

a = rasterParamGeneralized(
    aPerFrame = 512, frameRepeats =1,
    bscanPerBlock = 16, blockRepeats = 32, blockPerVolume = 8)
#print("frame size = " + str(a.frameSize(pixPerA)))


frameLocation = 30 # arbitrarily selected [0, bscanPerBlock * blockPervolume -1]


data = np.zeros((a.blockRepeats, a.aPerFrame, pixPerA), dtype='complex128')
for blockRepeatIndex in range(0, a.blockRepeats): # from 0 to blockRepeats - 1
    myOffsets = a.getFrameOffsets(frameLocation, pixPerA)
    frameSize = a.frameSize(pixPerA)
    tmpData = np.fromfile(file = filename, dtype='>f8',
                       count = frameSize*2,
                       offset = myOffsets[blockRepeatIndex]*pixByte) # *2 is to account for complex (f8 + f8)
    # reconstruct comple data from flow64 * 2
    tmpData = tmpData.reshape([frameSize,2])
    tmpData = tmpData[:,0] + 1j * tmpData[:,1]
    tmpData = tmpData.reshape(a.aPerFrame, pixPerA)
    data[blockRepeatIndex,:,:] = tmpData


linearData = np.abs(data)**2
logData = 20*np.log10(np.abs(data))
#LIV = np.nanvar(logData, axis = 0)
MeanIlog = np.nanmean(logData, axis = 0)
MeanIlin = 10*np.log10(np.nanmean(linearData, axis = 0))
#plt.imshow(logData[10,:,:])
#plt.imshow(np.transpose(LIV[:,300:750]), cmap='hsv', vmin = 0, vmax = 50)
plt.imshow(np.transpose(MeanIlin[:,300:750]), cmap='gray',vmin=50,vmax=80)

#plt.imshow(LIV)
#plt.imshow(10*np.log10(np.abs(data[10,:,:])))




# # bscanIndex = frameLocation % a.bscanPerBlock
# # blockIndex = int(frameLocation/a.bscanPerBlock)

# # singleBlockSize = a.frameRepeats * a.bscanPerBlock
# # fullBlockSize = singleBlockSize * a.blockRepeats
# # frameIndex = fullBlockSize * blockIndex + singleBlockSize * blockRepeatIndex + a.frameRepeats * bscanIndex

# # frameSize = a.aPerFrame * pixPerA # [pix]
# # theOffset = frameIndex*frameSize

# #fp = open(filename, "rb")
# #fp.seek(theOffset*pixByte)
# data = np.fromfile(file = filename, dtype='>f8', count = frameSize*2, offset = myOffset[0]*pixByte) # *2 is to account for complex (f8 + f8)
# # reconstruct comple data from flow64 * 2
# data = data.reshape([frameSize,2])
# data = data[:,0] + 1j * data[:,1]
# data = data.reshape(a.aPerFrame,pixPerA)

# plt.imshow(10*np.log10(np.abs(data)))
# #numpy.fromfile(file, dtype=float, count=- 1, sep='', offset=0, *, like=None)

#137 975 824 384