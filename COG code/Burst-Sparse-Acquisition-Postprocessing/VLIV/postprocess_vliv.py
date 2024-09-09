import numpy as np
import tifffile
from .variableLIV import *
from tqdm import tqdm
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import cv2
import time

debugger = False #put "pdb.set_trace()" in the location where you debug
if debugger == True:
    import pdb


def scale_clip(data, vmin, vmax, scale=1.0):
    return np.clip((data-vmin)*(scale/(vmax-vmin)), 0, scale)


def vliv_postprocessing(path_OCT, volumeDataType, frameSparationTime , 
                frameRepeat, bscanLocationPerBlock, blockRepeat, blockPerVolume, fitting_method = "GPU", 
                       mfInitial = 5, dfInitial = 10 , bounds = ([0,0],[np.inf, np.inf]), 
                       use_constraint = False , use_weight = False , average_LivCurve = False,
                       search_LivCurve_0 = False, search_LivCurve_noSaturate = False, motionCorrection = False,
                       octRange = (-5, 55), mfRange =(0, 10), dfRange =(0, 0.05)):
     
    """
    This function process VLIV using GPU-based curve fitting method
    
    Parameters
    ---------
    path_OCT: file path of linear OCT intensity
    volumeDataType: str
        Either "RealBSA" or 'Ibrahim2021BOE'
        RealBSA: Burst scanning protocol
        Ibrahim2021BOE: Dynamic OCT scanning protocol (Same as Ibrahim2021BOE paper)
    frameSparationTime: constant(float)
        Successive frame Measurement time [s] 
    frameRepeat: int
        Number of frames in a single burst
    bscanLocationPerBlock: int
        No of Bscan locations per Block
    blockRepeat:  int 
        Number of block repeats at each Bscan location
    blockPerVolume: int 
        Number of blocks in a volume
    fitting_method: str
        Either "CPU" or "GPU"
        CPU: vlivCPUFitExp()
        GPU: vlivGPUFitExp()
    mfInitial: float
        mfInitial = Initial value of a in fitting
    dfInitial: float
        1/dfInitial = Initial value of b in fitting
    bounds : 2D tuple
        bounds for fitting
        ([min a, min b], [max a, max b])
    use_constraint : True or False
        Set bounds of parameters in fitting : True
        don't set bounds : False
    use_weight : True or False
        Apply weight for fitting : True
        Don't apply weight : False
    average_LivCurve : True or False
        Average LIV curve before fitting using 2*2 or 3*3 kernel : True
        Don't average LIV curve : False
        
    Output
    -----------
    VLIV: 4D double [timePoints, slowscanlocations, z, x]
        LIV array at different time windows
    timewindows: 2D double
    Magnitude factor: 3D gray scale and RGB image
    Damping factor: 3D gray scale and RGB image
    
    """
    print("VLIV Processing started")
    maxFrameNumber = frameRepeat * blockRepeat * bscanLocationPerBlock # Maximum frame number per Block 
    burstingPeriod = frameRepeat *  bscanLocationPerBlock # [frames] # Peroid of burst sampling
    numLocation = bscanLocationPerBlock * blockPerVolume # Number of slow scan locations
    
    print('Processing: ' + path_OCT)
    # suffix_intensity_linear = "_OCTIntensityPDavg.tiff"
    suffix_intensity_linear = "abs2.tiff"
    root = path_OCT[:-len(suffix_intensity_linear)]
    path_vliv = root  +  '_vliv.tif'
    path_timewindow = root  +  '_timewindows.tif'
    path_vov = root + '_vov.tif'
    
    ## OCT intensity
    octFrames = tifffile.imread(path_OCT)
    height = octFrames.shape[1]
    width = octFrames.shape[2]
    
    if volumeDataType == 'RealBSA':
        print("Data Type = " + str(volumeDataType))
    
    elif volumeDataType == 'Ibrahim2021BOE':
        print("Data Type = " + str(volumeDataType))

    mf = np.zeros((numLocation, height, width), dtype=np.float32)
    df = np.zeros((numLocation, height, width),  dtype=np.float32)
    oct_db = np.zeros((numLocation, height, width),  dtype=np.float32)


    # Time_computeVLIV = 0
    # Time_fitting = 0
    # Start_compute = time.time()
    for floc in tqdm(range(0,numLocation)): #tqdm(range(0,numLocation))

        ## Load Sparse data
        if volumeDataType == 'RealBSA':
            sparseSequence, timePoints = makeSparseDataBSA (octFrames, maxFrameNumber, 
                                                            frameRepeat, burstingPeriod, bscanLocationPerBlock, floc)
        
        elif volumeDataType == 'Ibrahim2021BOE':
            sparseSequence, timePoints = makeSparseDataFromRasterRepeat(path_OCT, bscanLocationPerBlock, blockRepeat, floc)
        if motionCorrection == True:
            sparseSequence = correctMotion(sparseSequence, register_start_depth= 50)

        if floc == 0: #for save VLIV array
            VLIV_save = np.zeros((numLocation, timePoints.shape[0]-1, height, width),  dtype=np.float32)
            VoV_save = np.zeros((numLocation, timePoints.shape[0]-1, height, width),  dtype=np.float32)
        ## Compute logarithmic OCTs
        oct_db [floc] = 10*np.log10(np.nanmean(sparseSequence, axis=0))
        
        ## Compute VLIV
        # Start_computeVLIV = time.time()
        VLIV , possibleMtw , VoV = computeVLIV(sparseSequence, timePoints, maxTimeWidth =  np.nan, debug = 0)
        
        ## Average LIV curve
        if average_LivCurve == True:
            #VLIV : 3D array (time window, z, x)
            twIdx = 0
            for twIdx in range(0, VLIV.shape[0]):                
                VLIV[twIdx,:,:] = cv2.blur(VLIV[twIdx,:,:], (3,3))
                twIdx = twIdx + 1
        # End_computeVLIV = time.time()
        # Time_computeVLIV = Time_computeVLIV + (End_computeVLIV-Start_computeVLIV)
        # Start_fitting = time.time()
        ## curve fitting on VLIV curve and compute magnitude, damping factor
        if fitting_method == 'GPU':
            mag, damp = vlivGpuFitExp(VLIV, possibleMtw, VoV, frameSparationTime, mfInitial, dfInitial, bounds, use_constraint, use_weight)
        
        if fitting_method == 'CPU':
            mag, damp = vlivCPUFitExp(VLIV, possibleMtw, frameSparationTime, mfInitial, dfInitial, bounds, constraint = False)
        # End_fitting = time.time()
        # Time_fitting = Time_fitting + (End_fitting-Start_fitting)
        mf [floc] = mag ## Magnitude factor
        df [floc] = 1/ damp ## Damping factor
        VLIV_save[floc,:,:,:] = VLIV #for save VLIV array
        VoV_save[floc,:,:,:] = VoV

            
        if search_LivCurve_0 == True:
            compare = np.sqrt(np.mean(VLIV**2, axis = 0)) <= MSE[floc] #imprementation memo (spheroid data almost doesn't contain the pix of this case)
            mf[floc][compare] = 0
            df[floc][compare] = 0    
        
        if search_LivCurve_noSaturate == True:
            # damp: 2D map of time constant
            # mf[floc]: 2D map of Magnitude factor
            tau = damp # tau is the map of time constant
            mfimg = mf[floc] # 2D map of magnitude factor (alias)

            enableMap = np.ones(tau.shape)  * tau > 6.3*2 # True for missing pixel (fit failed pix)
            mfimg[enableMap] = np.nan
            mfimgNew = np.copy(mfimg)
            
            x_max = mfimg.shape[0]-1
            y_max = mfimg.shape[1]-1
            true_positions = np.where(enableMap)
            for x,y in zip(true_positions[0], true_positions[1]):
                if (x>=1 and x <= x_max-1 and y>=1 and y <= y_max-1):
                    newVal = np.nanmean(mfimg[x-1:x+2, y-1:y+2])
                    mfimgNew[x,y] = newVal
            mfimg = mfimgNew
            mfimg[mfimg==np.nan] = 0.0
            mf[floc] = mfimg
    End_compute = time.time()
    # print("VLIV computation  :  " + str(Time_computeVLIV) + "  s")
    # print("Fitting computation  :  " + str(Time_fitting) + "   s")
    # print(f'Computation time (from re-alighnment of frames to obtain ALIV/swiftness):{(End_compute-Start_compute)/60: .2f} min')
            #pdb.set_trace()


    tifffile.imwrite(root  +  '_dbOct.tif', oct_db )
    tifffile.imwrite(path_vliv, VLIV_save, append = True, contiguous=True)
    tifffile.imwrite(path_timewindow, possibleMtw, append=floc != 0)
    tifffile.imwrite(path_vov, VoV_save, append = True, contiguous = True)


    
    ## Convert to color magnitude and damping factor images
    mf_hsv = np.stack([scale_clip(mf, *mfRange, 0.33), np.ones_like(mf),
                       scale_clip(oct_db, *octRange)], axis=-1)
    mf_rgb = hsv_to_rgb(mf_hsv)

    df_hsv = np.stack([scale_clip(df, *dfRange, 0.33), np.ones_like(df),
                       scale_clip(oct_db, *octRange)], axis=-1)
    df_rgb = hsv_to_rgb(df_hsv)
    


    ## Save the gray scale and rgb images of magnitude and damping factor
    path_mag = root  +  '_MF.tif'
    path_mag_view = root + f'_MF_min{mfRange[0]}-max{mfRange[1]}.tif'

    path_damp = root  + '_DF.tif'
    path_damp_view = root + f'_DF_min{dfRange[0]}-max{dfRange[1]}.tif'


    tifffile.imwrite(path_mag, mf)  
    tifffile.imwrite(path_mag_view, (mf_rgb*255).astype(np.uint8)) 


    tifffile.imwrite(path_damp, df)
    tifffile.imwrite(path_damp_view, (df_rgb*255).astype(np.uint8))
    
    print("VLIV Processing Ended")