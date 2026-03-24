import numpy as np
import cupy as cp
import sys
import tifffile
import time
from .pygpufit import gpufit as gf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pdb
from skimage.registration import phase_cross_correlation
import scipy.ndimage as scim

def makeSparseDataBSA (octFrames, maxFrameNumber, singleBurstFrameNumber, burstingPeriod, lpg, floc):
    
    '''
    Create pseudo-sparse OCT sequence from burst sparse acquistion data.
    
    Parameters
    ---------
    octFrames: 3D double [timePoints, x, z]
        Time sequence OCT Intensity in linear scale.
    maxFrameNumber: int 
        The maximum number of frames in each block
        singleBurstFrameNumber: int 
        Number of frames in a single burst time acquisition block.
    burstingPeriod: int
        Time frame separation between the starting time points of two adjacent bursts.
    lpg: int
        Number of B-scan locations/group
    floc: int
        Frame location of interest
    
    Returns
    -----------
    sparseSequence: 3D double [timePoints, x, z]
        Time sequence OCT intensity under burst-sparse acquisition.
    timePoints: 1D int array
        The sequence of time points at which the sparse sequence were acquired
    
    '''
    block_no = np.int(floc/lpg)
    oneBurstTimePoints = np.array(range(0, singleBurstFrameNumber)) + (maxFrameNumber- burstingPeriod)*block_no
    timePoints = np.array(())
    
    for i in range(0, maxFrameNumber, burstingPeriod):
        timePoints = np.concatenate((timePoints, 4*floc + oneBurstTimePoints + i)).astype(np.int) 
    
    sparseSequence = np.array(octFrames[timePoints, :, :])
#     print(f"time points: { timePoints}")
    return sparseSequence, timePoints

def makeSparseDataFromRasterRepeat(inputFilePath, lpg, fpl, floc):
    """
    This function extracts time-sequence at a particular location from
    a raster-repeading data volume (Ibrahim2021BOE's volume).
    Parameters
    ----------
    OctInput : 3D double [timePoints, x, z] (or z, x?)
        Time sequnce OCT intensity taken by raster-repeating protocol (Ibrahim2021BOE).
    lpg : int
        Locations/group (here the group is "raster group." See Fig. 1 of Ibrahim 2021 BOE)
    fpl : int
        frames/location
    floc : int
        Frame Location of Interest.
        The data sequence at this location is returned as 'sparseSequence'.
        
    Returns
    -------
    sparseSequence : 3D double [timePoints, x, z] (or z, x?)
        Time sequence OCT intensity under speudo-sparse acquisition.
    timePoints : 1D int array in original frame index
        The sequence of time poins at which the sparseSequence was acquired.
    """

    fpg = lpg*fpl # frames/group
    theGrp = np.int(floc/lpg) # The group containing the location (loc)
    fIdxInG = floc - theGrp*lpg # The frame index in the group

    fStart = fpg * theGrp + fIdxInG # The start frame index in the volume of the location.
    fStop = fStart + (fpg - fIdxInG) - 1
    frameIndexes = range(int(fStart), int(fStop)+1, int(lpg))

    timePoints = np.linspace(0,fpl-1, fpl, dtype = np.int)*lpg
#     print(frameIndexes)
    with tifffile.TiffFile(inputFilePath) as imfile:
                OctInput = imfile.asarray()
                OctInput = OctInput[frameIndexes, ...]
    sparseSequence = np.array(OctInput)
    return(sparseSequence, timePoints)

def correctMotion(sparseSequence, register_start_depth):
    linearFrames = sparseSequence
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

    # Motion correction
    for fIndex in range(0,numFrames):
        # DataShifted[fIndex] = scim.shift(10*np.log10(linearFrames[fIndex]), shiftVal[fIndex])
        DataShifted[fIndex] = scim.shift(10*np.log10(linearFrames[fIndex]), shiftVal[fIndex])
        # change the line to db scale 
        # DataShifted[fIndex] = scim.shift(linearFrames[fIndex], shiftVal[fIndex])
    correctedSparseSequence = 10.0**(DataShifted/10.0)
    return(correctedSparseSequence)

def computeVLIV(OctSequence, timePoints, maxTimeWidth = np.nan, debug = 0):
    """
    compute VLIV from time sequential OCT linear intensity.

    Parameters
    ----------
    OctSequence : double (time, x, z) or (time, z, x)
        Time sequence of linear OCT intensity.
        It can be continuous sequence or sparse time sequence.
        
    timePoints : 1D int array (the same size with time of OctSequence)
        indicates the time point (in dt) at which the frames in the OCT sequence were taken.

    maxTimeWidth : int
        The maximum time width for LIV computation.
        If the LIV curve fitting uses only a particular time-region of LIV curve, 
        it is unnnessesary to compute LIV at the time region exceeding the fitting region.
        With this option, you can restrict the LIV computation time-region and
        can reduce the computation time.
        The default is NaN. If it is default, i.e., maxTimeWidth was not set,
        the full width will be computed.
        
    Returns
    -------
    VLIV : 3D double (max time window, x, z) or (max time window, z, x)

    possibleMtw : 1D int        
        Time points correnspinding to the max-time-window axis of VLIV.
        
    VoV : 3D double (max time window, x, z)
        variance of variances (LIVs) 
    """
    
    myName = sys._getframe().f_code.co_name
    
    # Compute all possible maximum time window
    timePoints = cp.asarray(timePoints)
    A = cp.ones((1,timePoints.shape[0]))
    B = cp.asarray(timePoints.reshape(timePoints.shape[0],1))
    timePointMatrix = cp.transpose(A*B) - A*B
    timePointMatrix[timePointMatrix<0] = 0.
    possibleMtw = cp.unique(timePointMatrix)
    possibleMtw = possibleMtw[possibleMtw != 0.] # to remove the elemnt of 0.0
    
    # Reduce the time-region to be computed to meet with "maxTimeWidth"
    if np.isnan(maxTimeWidth):
        pass
    else:
        maxTimeWidth = cp.asarray(maxTimeWidth)
        possibleMtw = possibleMtw[possibleMtw <= maxTimeWidth] 
    
    logSparseSequence = 10*np.log10(OctSequence)
    logSparseSequence = cp.asarray(logSparseSequence)             # for cupy computation
    VLIV = cp.zeros((possibleMtw.shape[0], logSparseSequence.shape[1], logSparseSequence.shape[2]))

    # variance of variance
    VoV = cp.zeros((possibleMtw.shape[0], logSparseSequence.shape[1], logSparseSequence.shape[2]))

    i = 0        
    for mtw in possibleMtw:
        if debug >= 2:
            startTime = time.time()
        
        validTimePointMatrix, trueMtw = seekDataForMaxTimeWindow(timePoints, mtw)
        validTimePointMatrix = cp.asarray(validTimePointMatrix)    # for cupy computation 
        #newly added for LIV for each subset
        Var = cp.zeros((validTimePointMatrix.shape[0], logSparseSequence.shape[1], logSparseSequence.shape[2] ))       
        # cupy compute
        for j in range(0, validTimePointMatrix.shape[0]):
            VLIV[i] = VLIV[i] + cp.nanvar(logSparseSequence[validTimePointMatrix[j,:]],axis=0)
            # newly added for LIV for each subset 
            Var[j] = cp.nanvar(logSparseSequence[validTimePointMatrix[j,:]],axis=0)

        VLIV[i] = VLIV[i] / (validTimePointMatrix.shape[0])
        
        #newly added for test, variance of variance
        # VoV at a single time window (2D array)
        VoV[i] =cp.var(Var, axis=0)
        
        i = i+1
        if debug >= 2:
            endTime = time.time()
            print("Time elapsed: "+str(endTime-startTime))
    # pdb.set_trace()
    VLIV = cp.asnumpy(VLIV)   # for numpy computation
    possibleMtw = cp.asnumpy(possibleMtw)
    VoV = cp.asnumpy(VoV)
    return(VLIV, possibleMtw, VoV)



def seekDataForMaxTimeWindow(timePoints, mtw):
    """
    Compute valid combination of timePoints for which the maximu timepoint
    is smaller than mtw. 
    Parameters
    ----------
    timePoints : 1D numpy array
        Sequence of time points at which OCT data was taken [in frame time unit]
    mtw : 1D array, int
        The set maximum time window [in fram time unit]

    Returns
    -------
    VTS: 2D numpy numpy array, bool.
        Valid time sequence. VTS[i,:] is the i-th valid time sequence.
    trueMtw: scholar int
        The real (true) time window size
    """
    A = cp.ones((1,timePoints.shape[0]))
    B = timePoints.reshape(timePoints.shape[0],1)
    timePointMatrix = cp.transpose(A*B) - A*B
    validTimePointMatrix = (timePointMatrix <= mtw)*(timePointMatrix >= 0)
    # for each [i,:]...
    # tureMtw = (maximum time point) of True - (minimum time point) of True
    # set to trueMtw
    
    ##---------------------
    ## Let's rewirte later as not to use for-loop
    ##---------------------    
    trueMtw = cp.zeros(validTimePointMatrix.shape[0])
    for i in range (0,validTimePointMatrix.shape[0]):
        X = validTimePointMatrix[i,:]
        X = timePoints[X]
        Y = cp.max(X) - cp.min(X)
        trueMtw[i] = Y
    
    # remove raws whcih are filled by only False
    ##validTimePointMatrix = validTimePointMatrix[(trueMtw > 0),:]
    ## trueMtw = trueMtw[trueMtw>0]
    validTimePointMatrix = validTimePointMatrix[(trueMtw >= mtw),:]
    trueMtw = trueMtw[trueMtw>= mtw]
    validTimePointMatrix = cp.asnumpy(validTimePointMatrix)
    trueMtw = cp.asnumpy(trueMtw)

    return(validTimePointMatrix, trueMtw)

def vlivGpuFitExp(VLIV, possibleMtw, VoV, frameSparationTime, mfInitial, dfInitial,
                  bounds = ([0,0],[np.inf, np.inf]), use_constraint = False, use_weight = False):
    """
    compute magnitude and damping factors from VLIV by exponential fitting.

    Parameters
    ----------
    VLIV : 3D double
        VLIV curve (time window, z, x)
    possibleMtw : 1d int
        time window indicators for VLIV data array.
    VoV : 3D double
        variance of varianves (LIVs) (time window, z, x)
    frameSparationTime: constant (float)
        Successive frame Measurement time [s] 
    bounds : 2D tuple
        ([min a, min b], [max a, max b])
    use_constraint : True or False
        Set bounds of parameters in fitting : True
        Don't set bounds : False
    use_weight : True or False
        Apply weight for fitting : True
        Don't apply weight : False
        

    Returns
    -------
    mag : 2d double (z, x)
        magnitude factor
    damp : 2d doubel (z, x)
    
    """
    # Roll the axis of VLIV such that the number of points should be at the last axis in order to fit using GPUfit
    VLIV = np.rollaxis(VLIV, 0, 3)
    height = VLIV.shape[0]
    width = VLIV.shape[1]
    n_points = VLIV.shape[2]
        
    # Reshape to 1D array 
    VLIV_re = np.reshape(VLIV,  (-1, n_points))
    # Number of fits 
    n_fits = height * width
    n_parameter = 2  
    
    # tolerance (smaller tolerance -> better fitting accuracy)
    tolerance = 1.0E-6
        
    # max_n_iterations
    max_n_iterations = 100
        
    # model id
    model_id = gf.ModelID.SATURATION_1D
        
    # initial parameters
    initial_parameters = np.ones((n_fits, n_parameter), dtype=np.float32)
    initial_parameters[:, 0] = mfInitial 
    # print(VLIV.shape)
    # initial_parameters[:, 0] = np.reshape(VLIV[:,:,-1], (n_fits))
    initial_parameters[:, 1] = 1/dfInitial
    
    if use_constraint == False : pass
    else :# boundary
        constraints = np.ones((n_fits, 2*n_parameter), dtype=np.float32) # parameters for gpufit_constrained()
        constraints[:,0] = bounds[0][0]
        constraints[:,1] = bounds[1][0]
        constraints[:,2] = bounds[0][1]
        constraints[:,3] = bounds[1][1]
        constraint_types = np.ones((n_parameter), dtype=np.int32)
        constraint_types[:] = 3
        
    if use_weight == False : pass
    else :# calculate weight for fitting based on VoV
    
        ## weight candudate-1 (variance of LIV)
        weight = 1/VoV # weight for fitting
        
        ## weight candidate-2 (data dependency corrected variaance of unbiased LIV)
        # NoS2V = possibleMtw/possibleMtw[0] + 1 # NoS2V : number of samples to compute Variance
        # VoUL = VoV * ((np.multiply(NoS2V, 1/(NoS2V-1)))**2)[:, np.newaxis, np.newaxis] # varianve of unbiased LIV
        # CVoUL = VoUL * (NoS2V/32)[:,np.newaxis, np.newaxis] # data dependency corrected variance of unbiased LIV (CVoUL)
        # weight = (1/CVoUL) # weight for fitting
        
        ## use one previous value instead of infinite value caused by the variance of a single LIV
        array_assign = weight[-2,:,:]
        weight[-1,:,:] = array_assign
        weight = (np.reshape(weight,  (-1, 31))).astype(np.float32) # Reshape to 1D array


        
        
    
    timeWindow = np.zeros(len(possibleMtw))
    for i in range (len(possibleMtw)):
        timeWindow[i] = possibleMtw[i] * frameSparationTime
        
    if use_constraint and use_weight : #use_constraint == True & use_weight == True
        parameters, states, chi_squares, number_iterations, execution_time =  gf.fit_constrained(np.ascontiguousarray(VLIV_re, dtype='float32'), weight , model_id, \
                                                                                          initial_parameters, constraints, constraint_types, tolerance, \
                                                                                          max_n_iterations, None, None, timeWindow.astype(np.float32))
    if use_constraint and not use_weight : #use_constraint == True & use_weight == False
        parameters, states, chi_squares, number_iterations, execution_time =  gf.fit_constrained(np.ascontiguousarray(VLIV_re, dtype='float32'), None, model_id, \
                                                                                      initial_parameters, constraints, constraint_types, tolerance, \
                                                                                      max_n_iterations, None, None, timeWindow.astype(np.float32))         
    if not use_constraint and use_weight : # use_constraint == False & use_weight == True 
        parameters, states, chi_squares, number_iterations, execution_time =  gf.fit(np.ascontiguousarray(VLIV_re, dtype='float32'), weight, model_id, \
                                                                                      initial_parameters, tolerance, \
                                                                                      max_n_iterations, None, None, timeWindow.astype(np.float32))
    if not use_constraint and not use_weight : #(default) use_constraint == False & use_weight == False
        parameters, states, chi_squares, number_iterations, execution_time =  gf.fit(np.ascontiguousarray(VLIV_re, dtype='float32'), None, model_id, \
                                                                                      initial_parameters, tolerance, \
                                                                                      max_n_iterations, None, None, timeWindow.astype(np.float32))
    


        
        

    mag = parameters [:, 0].reshape (height, width)
    damp = parameters [:, 1].reshape (height, width)
    
    return mag, damp


def vlivCPUFitExp(VLIV, possibleMtw, frameSparationTime, mfInitial, dfInitial,
                  bounds = ([0,0],[np.inf, np.inf]), constraint = False):
    """
    compute magnitude and damping factors from VLIV by exponential fitting.

    Parameters
    ----------
    VLIV : 3D double
        VLIV curve (time window, z, x)
    possibleMtw : 1d int
        time window indicators for VLIV data array.
    frameSparationTime: constant (float)
        Successive frame Measurement time [s] 
    bounds : 2D tuple
        bounds for fitting
        ([min a, min b], [max a, max b])
    constraint : True or False
        Set bounds of parameters in fitting : True
        don't set bounds : False

    Returns
    -------
    mag : 2d double (z, x)
        magnitude factor
    damp : 2d doubel (z, x)
    
    """
    height = VLIV.shape[1]
    width = VLIV.shape[2]
    mag = np.empty((height, width))
    damp = np.empty((height, width))

    def saturationFunction(x, a, b):
        return(np.absolute(a)*(1-(np.exp(-x/b))))
    
    for depth in range(0, np.int(height)):
      
        for lateral in range(0, np.int(width)):
            LivLine = VLIV[:,depth, lateral]
            # Remove nan from LivLine (and also from corresponding possibleMtw).
            nonNanPos = (np.isnan(LivLine) == False)
            if np.sum(nonNanPos) >= 2:
                LivLine2 = LivLine[nonNanPos]
                t = possibleMtw[nonNanPos]
                # 1D list of time window in frame units -> 1D list of time window in second unit.
                for i in range (len(t)):
                    t[i] = t[i] * frameSparationTime

                try:
                    if constraint == False:
                        popt, pcov = curve_fit(saturationFunction, 
                                           t,
                                           LivLine2,
                                            method = "lm", # when no boundary, "lm"
                                           p0 = [mfInitial, 1/dfInitial] )
                    else: 
                        popt, pcov = curve_fit(saturationFunction, 
                                           t,
                                           LivLine2,
                                            method = "dogbox",# when add boundary, "dogbox"
                                           p0 = [mfInitial, 1/dfInitial],
                                           bounds = ([0,0],[np.inf, np.inf]))# set boundary [min a, min b],[max a, max b]
                except RuntimeError:
#                     print("Fitting error occured at depth = "+str(depth) + " lateral = " + str(lateral))
                    mag[depth, lateral] = np.nan
                    damp[depth, lateral] = np.nan
                    
                mag[depth, lateral] = popt[0]
                damp[depth, lateral] = popt[1]

            else:
                mag[depth, lateral] = np.nan
                damp[depth, lateral] = np.nan
    
    return(mag, damp)
