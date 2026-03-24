# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:09:28 2023

@author: rionm
"""

import numpy as np
import cogCorrelation as corr
import time
from skimage.registration import phase_cross_correlation
import scipy.ndimage as scim
import tifffile

class octVolume:
    def __init__(self, filePath, rasterParamGeneralized, spectraPerA = 2048,
                 pixByte = 2, fileType = 'dat', systemType = None):
        """
            systemType: the identifyer of system from which spectraPerA and pixByte are automatically set.
                'Hyracotherium' for Hyracotherium SD-OCT.
        """
        self.filePath = filePath
        self.fileType = fileType
        self.rasterParam = rasterParamGeneralized
        self.systemType = systemType
        if self.systemType != None:
            self.pixByte = self.systemType2pixByte()
            self.spectraPerA = self.systemType2spectraPerA()
        else:
            self.pixByte = pixByte
            self.spectraPerA = spectraPerA
        # self.fileLoadFlag = False
        # self.motionCorrectedBIndexes = None
        # self.motionCorrectedFrames = None

    def systemType2pixByte(self):
        if self.systemType == 'Hyracotherium':
            return(2)
        else:
            return(None)
    
    def systemType2spectraPerA(self):
        if self.systemType == 'Hyracotherium':
            return(2048)
        else:
            return(None)

    def frameSpectra(self, bIndex, frameIndex = 0, frameRepeatIndex = 0):
        #myData = np.zeros((self.rasterParam.aPerFrame, self.spectraPerA), dtype='float')
        frameSize = self.rasterParam.aPerFrame * self.spectraPerA
        #
        # Need to be corrected !!!!!
        myOffset = frameSize * bIndex * self.pixByte

        frameSpectra = np.fromfile(file = self.filePath, dtype='>i2',
                              count = frameSize,
                              offset = myOffset )
        frameSpectra = frameSpectra.reshape(self.rasterParam.aPerFrame, self.spectraPerA)
        return(frameSpectra)
            
class rasterVolumeDummy:     
    def __init__(self, filePath, rasterParamGeneralized, pixPerA = 1024,
                 pixByte = 4, fileType = 'npy', signalScale = 'linear'):
        """
            signalScale = 'linear' or 'dB'
        """
        self.filePath = filePath
        self.fileType = fileType
        self.rasterParam = rasterParamGeneralized
        self.pixPerA = pixPerA
        self.pixByte = pixByte
        self.fileLoadFlag = False
        self.signalScale = signalScale
        self.motionCorrectedBIndexes = None
        self.motionCorrectedFrames = None
    
    def frameSequence(self, bIndex, outputSignalScale = 'dB', motionCorrection = None):
        """ 
            self.frameType =
            tiff: COG intensity tiff file (32-bit float, big endian)
            npy : NumPy binary.
            3dv : COG intensity 3dv file (32-bit float, big endian)
            NtuBin : Binary file (little endian) measured by NTC/CGU C++ measurement program/ CGU LabVIEW measurement program.
            CguBin : Binary file (big endian) measured by CGU LabVIEW measurement program.
        """
        if self.fileType == 'NtuBin' or self.fileType == 'CguBin' :
            #
            # For binary data loading
            #
            (frameIndexes, dummy) = self.rasterParam.getFrameIndexSequence(bIndex)
            myData = np.zeros((self.rasterParam.blockRepeats, self.rasterParam.aPerFrame, self.pixPerA), dtype='float32')
            j = 0
            for blockRepeatIndex in frameIndexes: # from 0 to blockRepeats - 1
                frameSize = self.rasterParam.frameSize(self.pixPerA) 
                file = self.filePath +"/"+str(blockRepeatIndex+1)+".bin"
                if self.fileType == 'NtuBin':
                    tmpData = np.fromfile(file , dtype=np.dtype('<f4'),
                       count=self.rasterParam.aPerFrame * self.pixPerA)
                elif self.fileType == 'CguBin':
                    tmpData = np.fromfile(file , dtype=np.dtype('>f4'),
                       count=self.rasterParam.aPerFrame * self.pixPerA)

                tmpData = tmpData.reshape(self.rasterParam.aPerFrame, self.pixPerA)
                myData[j,:,:] = tmpData    
                j = j+1
            myData = self.fixSignalScale(myData, 'linear')
            
        if self.fileType == 'tiff':
            #
            # For tiff data loading
            #
            (frameIndexes, dummy) = self.rasterParam.getFrameIndexSequence(bIndex)
            myData = np.zeros((self.rasterParam.blockRepeats, self.rasterParam.aPerFrame, self.pixPerA), dtype='float32')
            self.loadVolumeTiff() # This function load only if the tiff file was not loaded yet.
            myData = self.data[frameIndexes] #original
            myData = self.fixSignalScale(myData, 'linear')
            myData = myData.transpose(0, 2, 1)
            
        if self.fileType == 'npy':
            #
            # For npy data loading
            #
            (frameIndexes, dummy) = self.rasterParam.getFrameIndexSequence(bIndex)
            #print("Frame index: " + str(frameIndexes))
            myData = np.zeros((self.rasterParam.blockRepeats, self.rasterParam.aPerFrame, self.pixPerA), dtype='float32')
            self.loadVolumeNpy() # This function load only if the npy file was not loaded yet.
            myData = self.data[frameIndexes] #original
            myData = self.fixSignalScale(myData, 'linear')

        elif self.fileType == '3dv':
            #
            # For 3dv data loading
            #
            myData = np.zeros((self.rasterParam.blockRepeats, self.rasterParam.aPerFrame, self.pixPerA), dtype='float32')
            for blockRepeatIndex in range(0, self.rasterParam.blockRepeats): # from 0 to blockRepeats - 1
                (frameIndexes, dummy) = self.rasterParam.getFrameIndexSequence(bIndex)
                myOffsets = (np.array(frameIndexes, dtype = 'uint64') * self.pixPerA * self.rasterParam.aPerFrame * self.pixByte)
                frameSize = self.rasterParam.frameSize(self.pixPerA)
                tmpData = np.fromfile(file = self.filePath, dtype='>f4',
                                    count = frameSize,
                                    offset = int(myOffsets[blockRepeatIndex])) 
                tmpData = tmpData.reshape(self.rasterParam.aPerFrame, self.pixPerA)
                myData[blockRepeatIndex,:,:] = tmpData                
            myData = self.fixSignalScale(myData, 'linear')

        #
        # motion correction
        #
        # Buggy!!!! Carefully check linear-log issue
        #
        # Here we assume myData is linear
        if motionCorrection == 'testMethod':
            if self.motionCorrectedBIndexes != bIndex:
                linearFrames = myData
                
                DataShifted = np.zeros(linearFrames.shape)
                numFrames = linearFrames.shape[0]
                shiftVal = np.zeros((numFrames,2))

                algorithm = 'standard'
                if algorithm == 'standard':
                    # Motion detection                
                    for fIndex in range(0,numFrames):
                        #start_time_registration= time.time()
                        shift, error, diffphase = phase_cross_correlation(linearFrames[15], linearFrames[fIndex], upsample_factor=10)
                        #end_time_registration = time.time()
                        shiftVal[fIndex] = shift
                        # Option-1: Use log for interpolation
                    #start_time_shift= time.time()
                elif algorithm == 'roling':
                    print("roling method used.")
                    for fIndex in range(0,numFrames-1):
                        shift, error, diffphase = phase_cross_correlation(linearFrames[fIndex], linearFrames[fIndex+1], upsample_factor=10)
                        shiftVal[fIndex+1] = shift
                    shiftVal = np.cumsum(shiftVal, axis = 0)
                  
                
                # Motion correction
                for fIndex in range(0,numFrames):
                    DataShifted[fIndex] = scim.shift(10*np.log10(linearFrames[fIndex]), shiftVal[fIndex])
                myData = 10.0**(DataShifted/10.0)
                
                self.motionCorrectedBIndexes = bIndex
                self.motionCorrectedFrames = myData
            
            else:
                myData = self.motionCorrectedFrames
        
        myData = self.fixSignalScale(myData, outputSignalScale)
        return(myData, frameIndexes)

    def loadVolumeNpy(self):
        if self.fileLoadFlag == False :
            print("Npy loading...")
            self.data = np.load(self.filePath, mmap_mode='r')
            self.fileLoadFlag = True
            print("Npy loading done.")
            
    def loadVolumeBinary(self):
        if self.fileLoadFlag == False :
            print("Binary loading...")
            self.data = np.load(self.filePath, mmap_mode='r')
            self.fileLoadFlag = True
            print("Binary loading done.")
            
    def loadVolumeTiff(self):
        if self.fileLoadFlag == False :
            print("Tiff loading...")
            self.data = tifffile.imread(self.filePath)
            self.fileLoadFlag = True
            print("Tiff loading done.")
    
    def fixSignalScale(self, myData, outputSignalScale):
        """
        Pameters
        ----------
        myData : numpy float32 array
        outputSignalScale : 'dB' or 'linear'
        
        Returns
        -------
        numpy float32 in dB or linear scale.

        """
        # Set (change) the signal scale into 'linear' or 'dB'
        if self.signalScale == outputSignalScale:
            pass
        elif outputSignalScale == 'dB':
            myData = 10. * np.log10(myData)
        else : # outputSignalScale is 'linear' but current data scale is dB
            myData = 10.**(myData/10.)
        
        return(myData)

    def correctMotion(self, bIndex, outputSignalScale = 'dB'):
        pass
