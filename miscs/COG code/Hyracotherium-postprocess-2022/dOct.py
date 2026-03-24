import numpy as np
import cogCorrelation as corr
import time
from skimage.registration import phase_cross_correlation
import scipy.ndimage as scim
import tifffile

debugLevel = 2

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
    
    def getFrameOffsets_obsolete(self,frameLocation, pixPerA=2048):
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
    
    def getFrameIndexSequence(self, bIndex):
        # bIndex = 30
        # bPerBlock = 16
        # framePerB = 32
        framePerBlock = self.bscanPerBlock * self.blockRepeats
        blockIndex = int(bIndex / self.bscanPerBlock)
        bIdxInBlock = bIndex - blockIndex * self.bscanPerBlock
        firstFrame = blockIndex * framePerBlock + bIdxInBlock
        frameIndexes = range(firstFrame,firstFrame+framePerBlock, self.bscanPerBlock)
        return (list(frameIndexes), frameIndexes)
    
class rasterVolume:     
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
    
    def frameSequence(self, bIndex, outputSignalScale = 'dB', motionCorrection = None, register_start_depth = 0):
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
                linearFrames_short = linearFrames[:, : , register_start_depth:-1]
                DataShifted = np.zeros(linearFrames.shape)
                numFrames = linearFrames.shape[0]
                shiftVal = np.zeros((numFrames,2))

                algorithm = 'standard'
                if algorithm == 'standard':
                    # Motion detection                
                    for fIndex in range(0,numFrames):
                        #start_time_registration= time.time()
                        shift, error, diffphase = phase_cross_correlation(linearFrames_short[16], linearFrames_short[fIndex], upsample_factor=10)
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
                    # DataShifted[fIndex] = scim.shift(10*np.log10(linearFrames[fIndex]), shiftVal[fIndex])
                    DataShifted[fIndex] = scim.shift(10*np.log10(linearFrames[fIndex]), shiftVal[fIndex])
                    # change the line to db scale 
                    # DataShifted[fIndex] = scim.shift(linearFrames[fIndex], shiftVal[fIndex])
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
#            self.data = np.load(self.filePath)
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
            myData = 10.* np.log10(myData)
        else : # outputSignalScale is 'linear' but current data scale is dB
            myData = 10.**(myData/10.)
        
        return(myData)

    def correctMotion(self, bIndex, outputSignalScale = 'dB'):
        pass

def computeLivAndIntensity(dataVolume, bIndexes = (0), computeLIV = True, computeIntensity = True, motionCorrection = None, register_start_depth=0):
    """

    Parameters
    ----------
    dataVolume : rasterVolume.rasterVolume class 
        The object consisting the intensity data sequence.
        The data size and scan parameters are possees in the rasterVolume.rasterParamGeneralized object
        included in the dataVolume.
    bIndexes : the list-like object in which the indexes of B-scan to be processed is listed.
                The default is (0). In this case all the B-scans are processed.
        DESCRIPTION. The default is (0).
    computeIntensity = False
        If True, this method retruns dB-intensity volume as well as LIV.

    Returns
    -------
    (LIV, dB-intenity) if computeIntensity == True.
    (LIV, None) if False

    """
    myParam = dataVolume.rasterParam # Make alias for easy writing
    
    # Process each B-scan
    if bIndexes == (0):
        bIndexes = range(myParam.bscanPerBlock * myParam.blockPerVolume)
    else:
        pass

    if computeLIV == True:
        LIV = np.zeros((len(bIndexes), myParam.aPerFrame, dataVolume.pixPerA),dtype='float32')
    else:
        LIV = None
        
    if computeIntensity == True:
        logIntensity = np.zeros((len(bIndexes), myParam.aPerFrame, dataVolume.pixPerA),dtype='float32')
    else:
        logIntensity = None
        

    #
    # Compute LIV for each B-scan
    #    
    iterIndex = 0
    for bIndex in bIndexes:
             
        if (computeLIV or computeIntensity) == True:
            (logData, frameIndexes) = dataVolume.frameSequence(bIndex, outputSignalScale = 'dB', motionCorrection = motionCorrection, register_start_depth = register_start_depth)
            M = np.average(logData,axis=0) # Mean intensity of this B-scan.
    
        if computeLIV == True:
            Sub=np.zeros((logData.shape))
            for i in range(logData.shape[0]):
                Sub[i]=(logData[i]- M) 
            
            SUM=np.sum(Sub**2,axis=0)
            LIV[iterIndex]=np.divide(SUM,logData.shape[0])
    
        if computeIntensity == True:
            logIntensity[iterIndex] = M
        
        iterIndex += 1
        print(str(bIndex)+ ' frame location was processed.')
    
    return(LIV, logIntensity, logData)

def computeLiv(dataVolume, bIndexes = (0), motionCorrection = None, register_start_depth =0):
    """

    Parameters
    ----------
    dataVolume : rasterVolume.rasterVolume class 
        The object consisting the intensity data sequence.
        The data size and scan parameters are possees in the rasterVolume.rasterParamGeneralized object
        included in the dataVolume.
    bIndexes : the list-like object in which the indexes of B-scan to be processed is listed.
                The default is (0). In this case all the B-scans are processed.
        DESCRIPTION. The default is (0).
    Returns
    -------
    LIV
    """
    return( computeLivAndIntensity(dataVolume, bIndexes, computeLIV = True,
                                   computeIntensity = False, motionCorrection = motionCorrection, register_start_depth =0))

def computeMeanDbIntensity(dataVolume, bIndexes = (0), motionCorrection = None, register_start_depth =0):
    """

    Parameters
    ----------
    dataVolume : rasterVolume.rasterVolume class 
        The object consisting the intensity data sequence.
        The data size and scan parameters are possees in the rasterVolume.rasterParamGeneralized object
        included in the dataVolume.
    bIndexes : the list-like object in which the indexes of B-scan to be processed is listed.
                The default is (0). In this case all the B-scans are processed.
        DESCRIPTION. The default is (0).
    Returns
    -------
    dbIntensity (numpy 3d array, float)
    """
    return( computeLivAndIntensity(dataVolume, bIndexes, computeLIV = False,
                                   computeIntensity = True, motionCorrection = motionCorrection, register_start_depth =0))

def computeOcds(dataVolume, bIndexes = (0), frameSeparationTime = 12.8e-3,
                ocdsRanges = [(1,6)], computeDamp = False, motionCorrection = None, register_start_depth =0 ):
#def computeOcds(dataVolume, bIndexes = (0), frameSeparationTime = 12.8e-3, ocdsRanges=[(1,6)]):
    """
    Parameters
    ----------
    dataVolume : TYPE
        DESCRIPTION.
    bIndexes : TYPE, optional
        DESCRIPTION. The default is (0).
    frameSeparationTime : TYPE, optional
        DESCRIPTION. The default is 12.8e-3.
    ocdsRanges : ((OCDS1 fitting start, OCDS1 fitting end), (those for OCDS range 2), ...)
        List of tuples. Each tuple consists of two integers,
        which indicate the data point indexes of correlation curve, which are used for OCDS slope fitting.
        The default is ((1:6)). In this case one OCDS image is computed at the fitting range of 1st to 6th correlation points, where the non-delay correlation is at the 0th point.

    Returns
    -------
    None.

    """
    myParam = dataVolume.rasterParam # Make alias for easy writing
    
    # Process each B-scan
    if bIndexes == (0):
        bIndexes = range(myParam.bscanPerBlock * myParam.blockPerVolume)
    else:
        pass

    #
    # Compute LIV for each B-scan
    #
    if computeDamp == True:
        fieldMag = 2
    else:
        fieldMag = 1
        
    outImage = np.zeros((len(ocdsRanges) * fieldMag , len(bIndexes), myParam.aPerFrame, dataVolume.pixPerA),dtype='float32')
        
    bscanIterIndex = 0
    for bIndex in bIndexes: # Compute OCDS for each B-scan  # bIndexes = range(0, 125)
        if debugLevel >= 1:
            print("OCDS, B-scan # " + str(bIndex) + " is in process.")
        if debugLevel >= 2:
            start_time= time.time()
            
        (logData, frameIndexes) = dataVolume.frameSequence(bIndex, 
                                                           outputSignalScale = 'dB',
                                                           motionCorrection = motionCorrection,
                                                           register_start_depth = register_start_depth) # Load frame sequence at a particular B-scan
        cor=np.abs((corr.MaskedCorrelation1D(logData, logData, optLevel=2))) # Compute correlation curve as [Delay time, x, z]
        
        # Compute the delay time points (t) in the correlation curve (cor) in second.
        frameIndexes = np.array(frameIndexes, dtype='float')
        t = (frameIndexes - frameIndexes[0]) * frameSeparationTime

        # Fit correlation slope        
        imageTypeIndex = 0
        for fitRange in ocdsRanges:                    # ocdsRanges = [(1, 6)]
            corFit = cor[fitRange[0]:(fitRange[1]+1)]  # corFit = cor[1:7]
            tFit = t[fitRange[0]:(fitRange[1]+1)]      # tFit = t[1:7]
            # Least square fitting for correlation coefficient (corFit = X) and the correlation delay time (tFit = Y)
            X = np.transpose(tFit)
            Y = corFit
            aveXY = np.mean(np.transpose((X * np.transpose(Y))),0)
            aveX = np.mean(X)
            aveY = np.mean(Y,0)
            aveX2 = np.mean(X**2)
            corSlope = (aveXY - aveX*aveY)/(aveX2 - aveX**2)
            corIntercept = aveY - corSlope * aveX
            if debugLevel >= 5:
                print("shape: aveY: " +str(aveY.shape))
                print("shape: aveX: " +str(aveX.shape))
                print("shape: slope: " +str(corSlope.shape))
                print("shape: intercept: " +str(corIntercept.shape))
            outImage[imageTypeIndex * 2, bscanIterIndex] = -corSlope
            
            # For damping computation, see Note Yasuno2022-09:10
            t1 = int(frameIndexes[fitRange[0]]) * frameSeparationTime
            t2 = int(frameIndexes[fitRange[1]]) * frameSeparationTime
            print("t1: " +str(t1) + " t2: " +str(t2))
            dt1 = t2 - t1
            dt2 = t2**2. - t1**2.
            dt3 = t2**3. - t1**3.
            # print('imageTypeIndex= ', str(imageTypeIndex), 'bscanIterIndex=',
            #       str(bscanIterIndex), 'corSlope=',str(np.shape(corSlope)), 'corIntercept=', str(np.shape(corIntercept)),
            #       'dt1~dt3= ', str(dt1), str(dt2), str(dt3))
            # print('bIndexes is: ', str(bIndexes))
            # outImage[imageTypeIndex * 2+1, bscanIterIndex] = (2./3.)* (2.*corSlope*dt3 + corIntercept * dt2)/(corSlope*dt2 + 2*corIntercept* dt1)
            outImage[imageTypeIndex * 2 + 0, bscanIterIndex, :, :] = (2. / 3.) * (
                        2. * corSlope * dt3 + corIntercept * dt2) / (corSlope * dt2 + 2 * corIntercept * dt1)
            # imageTypeIndex=0; bscanIterIndex = 0; corSlope=[250,800]; corIntercept=[250,800]
            # dt1~dt3= [0.01,0.014, 0.0172]
            # outImage: [1,125,250,800]
            # outImage[0*2+1, index0] = ((2/3) *
            #                       (2*color_k * 0.0172|dt3| + color_b*0.014) /
            #                       (color_k*0.014|dt2| + 2*color_b*0.01|dt1|))
            imageTypeIndex += 1
        else: 
            pass
        
        # # Compute dumping coefficient
        # imageTypeIndex = 0
        # for fitRange in dumpFitRanges:
        #     corFit = cor[fitRange[0]:(fitRange[1]+1)]
        #     tFit = t[fitRange[0]:(fitRange[1]+1)]
        #     X = np.transpose(tFit)
        #     Y = corFit
        

        bscanIterIndex += 1   
        if debugLevel >= 2:
            print("   Computation time was " + str(time.time() - start_time))
            
    return(outImage)

