# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:05:08 2023

@author: rionm
"""

import os
import numpy as np
import sys
import tifffile
import time
from tqdm import tqdm
from VLIV.postprocess_vliv import *
from LIV. liv_postprocess import *
import matplotlib.pyplot as plt

import pdb
global tmpBuff


## Input file path of linear OCT intensity
path_OCT = [

            ]
volumeDataType =  "Ibrahim2021BOE"

liv_proc = False
vliv_proc = True

## Input parameters of the scanning protocols
frameRepeat = 1 # Number of frames in a single burst
bscanLocationPerBlock = 16 # No of Bscan locations per Block
blockRepeat =  32 # Number of block repeats at each Bscan location
blockPerVolume = 8 # Number of blocks in a volume


# frameSparationTime = 12.66e-3 # Successive frame Measurement time for Hyracotherium data[s] 
frameSparationTime = 12.8e-3 # Successive frame Measurement time for TransToad data[s] 

def main():
    for id in range(len(path_OCT)):
        tStart = time.time()
        
        if liv_proc:
            
            liv_postprocess (path_OCT,frameRepeat, blockPerVolume, 
                              bscanLocationPerBlock, blockRepeat, octRangedB= (10,60), LIVrange=(0, 10))
    
        if vliv_proc:
            
            vliv_postprocessing (path_OCT[id], volumeDataType, frameSparationTime,
                                frameRepeat, bscanLocationPerBlock, blockRepeat, blockPerVolume, fitting_method = "GPU",
                                mfInitial = 20, dfInitial = 1, bounds = ([0,0],[np.inf, np.inf]), 
                                use_constraint = True, use_weight = False, average_LivCurve = True,
                                search_LivCurve_0 = False, search_LivCurve_noSaturate = False, motionCorrection = False,
                                octRange = (10,40), mfRange =(0,40), dfRange =(0, 3))


        
        
    tEnd = time.time()
    print(f'Computation time:{(tEnd-tStart)/60: .2f} min')
    
    
if __name__ == "__main__":
    main()