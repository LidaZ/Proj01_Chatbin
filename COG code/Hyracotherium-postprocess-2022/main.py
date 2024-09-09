import numpy as np
import tifffile
import importlib
import dOct as doct
importlib.reload(doct)
import cog
importlib.reload(cog)
import imagecolorizer as icz
importlib.reload(icz)
import time
start_time_main = time.time()


inputFilePath = [r"F:\Data_2024\20240801_Jurkat02\lv_20hr\raster\Data_IntImg.tif"]

#
# set file type in "tiff", "npy", "3dv". And signal scale in "linear", "dB".
#
fileType = "tiff"
signalScale = "linear"

#
# setcorrmethord in "None", "testMethod"
#
setcorrmethord = "testMethod"
register_start_depth = 220   #limit the image depth for registration to avoid the artifacts

#
# set blur kernel size (blur kernel will be applied when colorization)
#
blurnumber = 1

#
# Set generalized raster parametrers
#
rastParam = doct.rasterParamGeneralized(
    aPerFrame=250,
    frameRepeats=1,  # 1
    bscanPerBlock=1, # 250
    blockRepeats=32,  # 1 `
    blockPerVolume=125)  # 1


frameSeparationTime = 0.02 # [s]
pixByte = 4  # for float32
pixPerA = 800 # sample point in A-line


# For each volume file
for dataId in range(0, len(inputFilePath)):
    filename = inputFilePath[dataId]

    # Create dataVolume and load the volume data from file into "dataVolume" object
    dataVolume = doct.rasterVolume( filePath=filename, rasterParamGeneralized=rastParam, pixPerA=pixPerA, pixByte=pixByte,
                                       fileType=fileType, signalScale=signalScale)
    
    myPath = cog.octFilePath(dataVolume.filePath)
    #
    # Add folder name if you make different folder to save
    #
    myPath.setOutRootDir("")
    
    #
    # set B-scan location to be process
    # if bIndexes = (0), all B-scan location will be processed.
    #
    bIndexes = (0)
    
    #
    #   Compute LIV and mean log intensity
    #
    (LIV, dbInt, logdata) = doct.computeLivAndIntensity(
        dataVolume, bIndexes=bIndexes, motionCorrection=setcorrmethord, register_start_depth=register_start_depth)
    
    LIV = np.swapaxes(LIV, 1, 2)
    dbInt = np.swapaxes(dbInt,1, 2) 
    np.save(myPath.makePath(f"_LIV_{setcorrmethord}", ".npy"), LIV )
    np.save(myPath.makePath(f"_dBInt_{setcorrmethord}", ".npy"), dbInt )
    
    #
    # Make pseudo color LIV + Intensity image
    #6
    inputRanges = [(0, 15), (-25, 20)]  # hyracotherium  (inputRanges = [(0., 10.), (-5., 55.)] #TransTold1)
    outputRanges = [(0, 120), (0.0, 1.0)]

    pcImage = icz.makeHVCompiteImage( H=LIV, V=dbInt, inputRanges=inputRanges, outputRanges=outputRanges, blurKernels=((blurnumber, blurnumber), (1, 1)) )
    tifffile.imwrite( myPath.makePath(f"_LIV_{setcorrmethord}_b{blurnumber}", ".tiff"), pcImage.astype("uint8"), photometric="rgb")#, compress=6)
    # save grayscale LIV 
    pcImage = icz.makeGrayImage(Vol=LIV, inputRange=inputRanges[1])
    tifffile.imwrite( myPath.makePath(f"_grayLIV_{setcorrmethord}_b{blurnumber}", ".tiff"), pcImage.astype("uint8"), photometric="rgb")#, compress=6)
    # save grayscale dB-scale intensity
    pcImage = icz.makeGrayImage(Vol=dbInt, inputRange=inputRanges[1])
    tifffile.imwrite( myPath.makePath(f"_dBint_{setcorrmethord}_b{blurnumber}", ".tiff"), pcImage.astype("uint8"), photometric="rgb")#, compress=6)

    #
    #   Compute OCDS and damp (When you don't process OCDS, please comment out the following lines.)
    #
    Ocds_full = doct.computeOcds(dataVolume, bIndexes = bIndexes, ocdsRanges = [(1,6)], computeDamp = False, frameSeparationTime = frameSeparationTime,
                                 motionCorrection = setcorrmethord, register_start_depth = register_start_depth) 
    OCDS = Ocds_full[0]
    # Damp = Ocds_full[1] # if computeDamp = True, please add thie line.
    OCDS = np.swapaxes(OCDS, 1, 2)
    np.save(myPath.makePath(f'_OCDSl_{setcorrmethord}', '.npy'), OCDS)
    # np.save(myPath.makePath(f'_damp_{setcorrmethord}_b{blurnumber}', '.npy'), Damp) # if computeDamp = True, please add thie line.
    
    #
    # Make pseudo color OCDS image
    #
    inputRanges = [(0, 0.6), (-25., 20.)]
    outputRanges = [(0., 120), (0., 1.)]
    pcImage = icz.makeHVCompiteImage(H = OCDS, V = dbInt, inputRanges =  inputRanges, outputRanges = outputRanges, blurKernels = ((blurnumber,blurnumber), (1,1)))
    tifffile.imwrite(myPath.makePath(f'_OCDSl_{setcorrmethord}_b{blurnumber}', '.tiff'), pcImage.astype('uint8'), photometric='rgb')#,compress=6)

    # Make pseudo color Damp coefficient image # if computeDamp = True, please add the following lines.
    # inputRanges = [(0., 50.), (-5.0, 55.0)]
    # outputRanges = [(0., 120), (0., 1.)]
    # pcImage = icz.makeHVCompiteImage(H = Damp, V = dbInt,
    #                                   inputRanges =  inputRanges, outputRanges = outputRanges,
    #                                   blurKernels = ((blurnumber,blurnumber), (1,1)))
    # tifffile.imsave(myPath.makePath(F'_damp_{setcorrmethord}_b{blurnumber}', '.tiff'), pcImage.astype('uint8'), photometric='rgb',compress=6)
    
    print( "whole processing time (for one data) is  " + str(-(start_time_main - time.time())) + "s")

