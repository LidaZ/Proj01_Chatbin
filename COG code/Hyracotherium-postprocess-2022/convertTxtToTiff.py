# -*- coding: utf-8 -*-

import time
start = time.time()
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from glob import glob
import os
from natsort import natsorted
from tifffile import imsave
import colorsys
import csv
import sys

#definition of the function to read the .txt files
def read_all_files(x):
   file_names = natsorted(glob(x))[0:350] #only 350 frames are selected from the 2000 farmes 
   files = [np.loadtxt(f) for f in file_names]
   OCT = np.concatenate(files,axis=0).reshape(350,1000,500) # (depth is 350,1000, 500 for Tai-Ang data, 350,1000,600 for Ken data) import data shape here (frames, depth, Alines)
   return(OCT)  

#.....main code......#

path_input_array1 = [
                
                    # r"D:\testData_Taiwan\Eraser_Sun\Process TXT File",
                   # r"F:\CGU\Processed\eraser",
                   r"D:\testData_Taiwan\Eraser_Sun_linear\Process TXT File"
                   
                   
                    ]


num_bscan = [350]  

plt.close("all")

for j in range(len(path_input_array1)):

    LinearOCT=np.array(read_all_files(str(path_input_array1[j]+"/*")), dtype='f4')
    LinearOCT_rot=np.array([np.rot90( LinearOCT[i],k=3) for i in range(len(LinearOCT))]) # rotate the inetsity to the correct order
    root =path_input_array1[j]
    print ("root = {}".format(root))
    imsave(root + '_FloatOCT.tif', LinearOCT_rot.astype(dtype='f4')) # save the linear OCT as stack of images

