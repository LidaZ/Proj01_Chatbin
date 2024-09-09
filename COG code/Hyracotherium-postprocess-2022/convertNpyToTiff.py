# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:30:54 2022

@author: rionm
"""
import numpy as np
import os
import tifffile

data_path = r"G:\20220510\lung_004_abs2_LIV.npy"

arr = np.load(data_path)

root = os.path.splitext(data_path)[0]

tifffile.imsave(root+ '.tif', arr.astype(dtype='f4'))