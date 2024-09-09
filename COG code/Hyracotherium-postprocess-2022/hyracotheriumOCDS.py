# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:23:03 2022

@author: rionm
"""

#this program is based on 3-D_OCDS_KernelSummationBased_LSF.py (Ibrahim's program).

import time
start_time= time.time()
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from tifffile import imsave
from scipy import signal
from scipy import ndimage
from scipy.stats import linregress
import sys
import copy
import colorsys
import csv

def save_array_as_csv(input_array, output_path):
    """
    This method saves 1D or 2D array as csv file.
    Args:
        input_array: 1D or 2D array to save as csv (1D or 2D array of float)
        output_path: output file path (str)
    Returns:
    """
    ##### set parameters
    num_column = np.shape(input_array)[1]
    num_row = np.shape(input_array)[0]
    
    ##### initialize list
    list_tofile = [[0 for i in range(num_column)] for j in range(num_row)] # initialize list
    
    ##### substitute values in array to list
    for i in range(num_row):
        list_tofile[i] = input_array[i]
    
    ##### write list
    with open(output_path, 'wb') as f:
        writer = csv.writer(f)
        for row in list_tofile:
            writer.writerow(row)
            
def readCSV(fn):
    ##### read list
    csv_list = []
    with open(fn, 'rb') as csvfile:
    #with open(dir_script + '\list.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i in csvfile:
            i = "".join(i.split('\r\n')) # remove '\r\n' (new line) from i
            i = i.split(",") # index column
            csv_list.append(i) # append row
    #print "csv_list = " + str(csv_list)
    
    max_num_column = len(csv_list[0])
    #print "len(csv_list[0]) = " + str(len(csv_list[0]))
    if len(csv_list) != 1:
        for row in range(1, len(csv_list)):
            #print "row = " + str(row)
            #print "len(csv_list[row]) = " + str(len(csv_list[row]))
            #print "len(csv_list[row-1]) = " + str(len(csv_list[row-1]))
            if len(csv_list[row]) > len(csv_list[row - 1]):
                max_num_column = len(csv_list[row])
            else: pass
    else: pass
        
    num_column = 2
    #print "max_num_column = " + str(max_num_column)
    if max_num_column > num_column:
        for row in range(len(csv_list)):
            csv_list[row] = csv_list[row][:num_column]
        max_num_column = num_column
    else: pass
    
    list_io = [["" for i in range(num_column)] for j in range(len(csv_list))]
    for row in range(len(csv_list)):
        for column in range(max_num_column):
            list_io[row][column] = csv_list[row][column]
    #print "list_io = " + str(list_io)
    
    l_si_i = list_io[1:][:]
    #print "len(l_si_i) = " + str(len(l_si_i))
    for row in range(len(csv_list)-1):
        for column in range(max_num_column):
            l_si_i[row][column] = int(l_si_i[row][column])
    #print "l_si_i = " + str(l_si_i)
    
    return l_si_i
                                               
def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in  colorsys.hsv_to_rgb(h,s,v))
    
def hsv_to_rgb(H, S, V):
        """
        Converts HSL color array to RGB array
    
        H = [0..360]
        S = [0..1]
        V = [0..1]
    
        http://en.wikipedia.org/wiki/HSL_and_HSV#From_HSL
        
        Arguments:
            H: Hue
            S: Saturation
            V: Value (brightness)
    
        Returns:
            R, G, B in [0..255]
        """
        # chroma
        C = V * S
        
        # H' and X (intermediate value for the second largest component)
        Hp = H / 60.0
        X = C * (1 - np.absolute(np.mod(Hp, 2) - 1))
    
        # initilize with zero
        R = np.zeros(H.shape, float)
        G = np.zeros(H.shape, float)
        B = np.zeros(H.shape, float)
    
        # handle each case:
        mask = (Hp >= 0) == ( Hp < 1)
        R[mask] = C[mask]
        G[mask] = X[mask]
    
        mask = (Hp >= 1) == ( Hp < 2)
        R[mask] = X[mask]
        G[mask] = C[mask]
    
        mask = (Hp >= 2) == ( Hp < 3)
        G[mask] = C[mask]
        B[mask] = X[mask]
    
        mask = (Hp >= 3) == ( Hp < 4)
        G[mask] = X[mask]
        B[mask] = C[mask]
    
        mask = (Hp >= 4) == ( Hp < 5)
        R[mask] = X[mask]
        B[mask] = C[mask]
    
        mask = (Hp >= 5) == ( Hp < 6)
        R[mask] = C[mask]
        B[mask] = X[mask]
        
        # adding the same amount to each component, to match value
        m = V - C
        R += m
        G += m
        B += m
        
        # [0..1] to [0..255]
        R *= 255.0
        G *= 255.0
        B *= 255.0
    
        return R.astype(int), G.astype(int), B.astype(int)
        
def combine_rgb_arrays(self, r, g, b):
        """
        Combine RGB arrays (support N-dimensional array)
        
        Arguments:
            r: ndarray which contains R-channel values
            g: ndarray which contains G-channel values
            b: ndarray which contains B-channel values
    
        Returns:
            rgb: (n+1)d-array which contains all three channel values. The final dimension has three elements which corespond to RGB values. 
        """
        ### combine RGB arrays
        temp = np.concatenate([[r], [g], [b]], axis = 0)
        rgb = np.rollaxis(temp, 0, len(np.shape(r))+1).astype('uint8')
        
        return rgb
             
#correlation functions 
debugFlag = 1
#defenition of the cross correlation function for 3D Input data
def cogCrossCorrelationNonNormalized (X,Y):
    if (X.shape != Y.shape) :
        print("X and Y must have the same shape.")
        return(0)
    
    # Add zeros at the end of the data-sequence.
    # This is to avoid aliasing of the data during FFT.
    Z = np.zeros(X.shape)
    tLength = Z.shape[0]
    XX = np.concatenate((X,Z),axis=0)
    XX_ft = np.fft.fft(XX,axis=0)

    # Branch process for cross-correlation and auto-correlation.
    # If X == Y (i.e. auto-correlation), the process can be accelarated by
    # skiping one FFT operation.
    if X is Y:
        YY_ft = XX_ft
        if debugFlag >= 1:
            print("Optimized code used.")
    else:
        YY = np.concatenate((Y,Z),axis=0)
        YY_ft = np.fft.fft(YY,axis=0)
        
    
    CC = np.fft.ifft(np.multiply(np.conj(XX_ft),YY_ft), axis=0)
    CC = CC[0:tLength]
    return(CC)  

### This function is obsolete
def cogCrossCorrelationNonNormalized_nonOptimized_dontUse (X,Y):
    if (X.shape != Y.shape) :
        print("X and Y must have the same shape.")
        return(0)
    Z = np.zeros(X.shape)
    tLength = Z.shape[0]
    XX = np.concatenate((X,Z),axis=0)
    YY = np.concatenate((Y,Z),axis=0)
    XX_ft = np.fft.fft(XX,axis=0)
    YY_ft = np.fft.fft(YY,axis=0)
    CC = np.fft.ifft(np.multiply(np.conj(XX_ft),YY_ft), axis=0)
    CC = CC[0:tLength]
    return(CC)  
      
def cogCrossCorrelationNormalized (X,Y):
    CC = cogCrossCorrelationNonNormalized(X,Y)
    DcSig = np.ones((X.shape[0],1,1))
    Denom = cogCrossCorrelationNonNormalized(DcSig,DcSig)
    CC= np.divide(CC,Denom)
    return(CC)  

def cogAutoCorrelationNonNormalized (X):
    # Compute auto-correlation function (numerator of correlation coefficient)
    AC = cogCrossCorrelationNonNormalized(X,X)
    return(AC)
    
def cogAutoCorrelationNormalized (X):
    
    AC=cogCrossCorrelationNormalized(X,X)
    return(AC)
#fast cross correlation function   
def FastCorrWithMask(F, maskfirst=True):
    """
    This function compute the cross correlation function between
    F (1-3 D array) with 1D rectangle mask.
    maskFirst = ture, for M (*) F where M is the mask, (*) is correlation operation.
        It is (hopefully) equivalent with cogCrossCorrelationNonNormalized(M, F)
    maskFirst = false, for F (*) M where M is the mask, (*) is correlation operation.
        It is (I believe) 
        equivalent with cogCrossCorrelationNonNormalized(F, M)
    """
    Fall = np.sum(F,axis=0)
    if maskfirst is True:
        Fcumsum = np.nancumsum(F,axis=0)
    else:
        Fcumsum = np.nancumsum(F[::-1],axis=0)
    zeroarray = np.zeros(F[0].shape)
    Fcumsum_conc=np.insert(Fcumsum,0,zeroarray,axis=0)[:-1,:,:]
    result = np.subtract( Fall,Fcumsum_conc)
    return(result)
    

     
 # def cog1DMaskedCorrelation_slow(F,G):
#    Mf = np.ones((F.shape))
#    Mg = np.ones((G.shape))
#    fmf_gmg = cogCrossCorrelationNonNormalized(F, G)
#    fmf_mg  = cogCrossCorrelationNonNormalized(F, Mg)
#    mf_gmg  = cogCrossCorrelationNonNormalized(Mf,G)
#    mf_mg   = cogCrossCorrelationNonNormalized(Mf,Mg)
#    fmf2_mg  = cogCrossCorrelationNonNormalized(np.square(F),Mg)
#    mf_gmg2  = cogCrossCorrelationNonNormalized(Mf,np.square(G))
#    Numerator = np.subtract (fmf_gmg, np.divide(np.multiply(fmf_mg,mf_gmg),mf_mg))#
#    Denominator = np.sqrt(np.multiply(np.subtract(fmf2_mg,np.divide(np.square(fmf_mg),mf_mg)), np.subtract(mf_gmg2,np.divide(np.square(mf_gmg),mf_mg)))) 
#    CorrCoef_Slow= np.divide(Numerator, Denominator)
#    return(CorrCoef_Slow)

##### FAST
def cog1DMaskedCorrelation(F,G, optLevel=2):    
    Mf = np.ones((F.shape))
    Mg = np.ones((G.shape))
    fmf_gmg = cogCrossCorrelationNonNormalized(F, G)
    if optLevel >= 2:
        fmf_mg = FastCorrWithMask(F, maskfirst=False)
        mf_gmg = FastCorrWithMask(G, maskfirst=True)
    else:
        fmf_mg  = cogCrossCorrelationNonNormalized(F, Mg)
        mf_gmg  = cogCrossCorrelationNonNormalized(Mf,G)
        
    # Compute mf_mg in a optimal way.
    if optLevel >= 1:
        # Fast algorithm
        mf_mg = cogCrossCorrelationNonNormalized(Mf[:,0,0], Mg[:,0,0])
        mf_mg = mf_mg.reshape(F.shape[0],1,1)
        if debugFlag >= 1:
            print("Optimal for mf_mg")
    else :
        #Slow but established method
        mf_mg   = cogCrossCorrelationNonNormalized(Mf,Mg)
        if debugFlag >= 1:
            print("Non Optimal for mf_mg")

    if optLevel >= 2:
        fmf2_mg = FastCorrWithMask(np.square(F), maskfirst=False)
        mf_gmg2 = FastCorrWithMask(np.square(G), maskfirst=True)
    else:
        fmf2_mg  = cogCrossCorrelationNonNormalized(np.square(F),Mg)
        mf_gmg2  = cogCrossCorrelationNonNormalized(Mf,np.square(G))
    
    cov=np.divide((np.subtract(fmf_gmg,np.divide(np.multiply(fmf_mg,mf_gmg),mf_mg))),mf_mg) #eq30 Checn2017
    KNL=np.ones((1,2,4))
    # COV=ndimage.convolve(np.real(cov),KNL, mode='constant')
    #splitting real and imag and collect them again
    cov_real=ndimage.convolve(np.real(cov),KNL, mode='constant')
    cov_imag=ndimage.convolve(np.imag(cov),KNL, mode='constant')
    COV=cov_real+1j*cov_imag
    
    var1=np.divide((np.subtract(fmf2_mg,(np.divide(np.square(fmf_mg ),mf_mg)))),mf_mg)   #eq31 Checn2017
    # VAR1=ndimage.convolve(np.real(var1),KNL, mode='constant')
    #splitting real and imag and collect them again
    var1_real=ndimage.convolve(np.real(var1),KNL, mode='constant')
    var1_imag=ndimage.convolve(np.imag(var1),KNL, mode='constant')
    VAR1=var1_real+1j*var1_imag
    
    var2=np.divide((np.subtract(mf_gmg2,(np.divide((np.square(mf_gmg)),mf_mg)))),mf_mg) #eq32 Checn2017
    # VAR2=ndimage.convolve(np.real(var2),KNL, mode='constant')
    
    #splitting real and imag and collect them again
    var2_real=ndimage.convolve(np.real(var2),KNL, mode='constant')
    var2_imag=ndimage.convolve(np.imag(var2),KNL, mode='constant')
    VAR2=var2_real+1j*var2_imag
    
    Denominator=np.multiply(np.sqrt(VAR1),np.sqrt(VAR2))
    
    # Numerator = np.subtract (fmf_gmg, np.divide(np.multiply(fmf_mg,mf_gmg),mf_mg))
    # Denominator = np.sqrt(np.multiply(np.subtract(fmf2_mg,np.divide(np.square(fmf_mg),mf_mg)), np.subtract(mf_gmg2,np.divide(np.square(mf_gmg),mf_mg)))) 
    CorrCoef_Fast= np.divide(COV, Denominator)
    return(CorrCoef_Fast)
#--------- Main Test Code -------
path_input_array1 = [
r"D:\programs\Hyracotherium-postprocess-2022\vibration noise correction\data\20220908\tmp1\047_abs2_corrected.npy"
#r"E:\20220610\005_abs2.3dv"
# r"N:\20220414_data\MCF7_Spheroid_003\MCF7_Spheroid_20220414_003_OCTIntensityPDavg.tiff",
# r"N:\20220414_data\MCF7_Spheroid_008\MCF7_Spheroid_20220414_008_OCTIntensityPDavg.tiff",
# r"N:\20220414_data\MCF7_Spheroid_013\MCF7_Spheroid_20220414_013_OCTIntensityPDavg.tiff",
# r"N:\20220414_data\MCF7_Spheroid_018\MCF7_Spheroid_20220414_018_OCTIntensityPDavg.tiff",
# r"N:\20220414_data\MCF7_Spheroid_023\MCF7_Spheroid_20220414_023_OCTIntensityPDavg.tiff",

  ] # absoluted & squared intensity, float32 , (all B-scan such as 4096, pixPerA, aPerB)
inputFileType = 3 # tiff = 1, 3dv = 2, numpyarray = 3
                
path_input_array2= [
# r"N:\20220414_data\MCF7_Spheroid_003\MCF7_Spheroid_20220414_003_OCTIntPDavg_view.tif",
# r"N:\20220414_data\MCF7_Spheroid_008\MCF7_Spheroid_20220414_008_OCTIntPDavg_view.tif",
# r"N:\20220414_data\MCF7_Spheroid_013\MCF7_Spheroid_20220414_013_OCTIntPDavg_view.tif",
# r"N:\20220414_data\MCF7_Spheroid_018\MCF7_Spheroid_20220414_018_OCTIntPDavg_view.tif",
# r"N:\20220414_data\MCF7_Spheroid_023\MCF7_Spheroid_20220414_023_OCTIntPDavg_view.tif",

  ] # dB-scale & [0,255] intensity, unit8, (all B-scan such as 4096, pixPerA, aPerB)
                                        
flag_color_composition = 0 # default = 0, if you process data until color composition = 1                                       
                  
pixPerA = 1024
aPerB = 512         
pixByte = 4
framePerVolume = 4096
frameSize = pixPerA*aPerB          
B_scans_used_for_OCDS= [32] #[4,8,16,32,64]   #[8,16,32,64]   !!!!#change the number of B scans herez
fitting_start=1
fitting_end=6
max_time_window=511 #set the b scan number at the maximum of the time window -1

real_time_window_value_in_sec= (max_time_window+1) * 0.0128

for j in range(len(path_input_array1)):
    root = os.path.splitext(path_input_array1[j])[0]
    print ("Processing...{}".format(root))
    ### Parameters this number refers to the image number you want to display between 0 to 511
    ##### Read data (tiff file)
    if inputFileType ==1:
        with tifffile.TiffFile(path_input_array1[j]) as imfile:
            input_array1 = imfile.asarray() 
            (bPerC, ptPerA, aPerB) = np.shape(input_array1)
            print ("(bPerC, ptPerA, aPerB) = " + str((bPerC, ptPerA, aPerB)))
    #####
    ##### Read data (3dv file)
    if inputFileType == 2:
        Data = np.zeros((framePerVolume, pixPerA,aPerB), dtype='float32')
        for frameId in range (0, framePerVolume):
            tmpData = np.fromfile(file = path_input_array1[j], dtype='>f4', 
                                      count = frameSize, offset = frameSize*frameId*pixByte)
            tmpData = tmpData.reshape(aPerB, pixPerA)
            tmpData = np.transpose(tmpData)
            Data[frameId,:,:] = tmpData  
        input_array1 = Data
        (bPerC, ptPerA, aPerB) = np.shape(input_array1)
        print ("(bPerC, ptPerA, aPerB) = " + str((bPerC, ptPerA, aPerB))) 
    #####
    ##### Read data (numpy file)
    if inputFileType == 3:
        input_array1 = np.zeros((framePerVolume, pixPerA,aPerB), dtype='float32')
        input_array1 = np.load(path_input_array1[j])
        (bPerC, ptPerA, aPerB) = np.shape(input_array1)
        print ("(bPerC, ptPerA, aPerB) = " + str((bPerC, ptPerA, aPerB))) 
    #####
        
    
    ##### OCDS calculations 

    for k in range(len(B_scans_used_for_OCDS)):
        num_frame = B_scans_used_for_OCDS[k]
        print("num frame: " + str(num_frame))
        num_slow_scan_location = bPerC / num_frame
        num_slow_scan_location_per_unit = (max_time_window +1) / num_frame
        num_unit = bPerC / (max_time_window +1)
        
        ###Process volume (for each slow scan location)
        Corrcoef_volume = np.array([]) # Initialize array (bPerC, ptPerA, aPerB)

       
        
        D = np.zeros((num_frame,ptPerA,aPerB)) # Initialize array (Frame, ptPerA, aPerB)
        for u in range(int(num_unit)): # for each block?
            #print "u = " + str(u)
            for l in range(int(num_slow_scan_location_per_unit)): # for each B-scan location in the block?
                bscan_id =np.arange (0, (max_time_window+1), (((max_time_window+1) / num_frame))) +l + (u * (max_time_window +1))
                #print "bscan_id = " + str(bscan_id)
                
                for i in  range(len(bscan_id)):
                    D[i] = 10*np.log10(input_array1[int(bscan_id[i])], dtype = "f4")
                cor=np.abs((cog1DMaskedCorrelation(D, D, optLevel=2)))
                
                Corrcoef_volume=np.append(Corrcoef_volume,cor)
                print("l : " + str(l) + " Per 16, u : " +str(u) + " Per 8, k: "+str(k) + " Per 1")
        Corrcoef_volume=np.reshape(Corrcoef_volume,(framePerVolume,pixPerA,aPerB)) #(Corrcoef_volume,(4096,402,512))
        root = os.path.splitext(path_input_array1[j])[0]
        print ("root = {}".format(root))
        imsave(root + '_CorCoef' +str([B_scans_used_for_OCDS[k]])+ str(real_time_window_value_in_sec)+ '2_4Kernelsum_abs.tif', Corrcoef_volume.astype(dtype='f4')) 
        
##############lest square fitting function 
    Corrcoef_volume=np.array(Corrcoef_volume)
    num_frame = B_scans_used_for_OCDS[0]
    num_location = int(bPerC / num_frame) # 128
    print ("num_location = " + str(num_location))
    x = np.arange(0,6553.6,6553.6/num_frame)
    t_start1 = time.time()
    OCDS_volume = np.zeros((num_location, ptPerA, aPerB))
    temp = x[fitting_start:fitting_end +1]
    t_2dim = np.tile(temp, (ptPerA, 1))
    for n in range(0, bPerC, num_frame):
        t_start2 = time.time()
        location = int(n / num_frame)
        print ("Location " + str(location) + " is being processed.")
        speed_map=np.zeros((ptPerA, aPerB)) 
        for X in range(aPerB):#row index
            Corrcoef_2dim = Corrcoef_volume[n:n+(num_frame-1)][fitting_start:fitting_end +1, :,X]
            Corrcoef_2dim = Corrcoef_2dim.T
            num = np.full((t_2dim).shape[0], (t_2dim).shape[1]) # (Shape, Value)
            slope = ((np.diag(np.dot(t_2dim, Corrcoef_2dim.T))- np.sum(Corrcoef_2dim, axis=1) * np.sum(t_2dim, axis=1)/num)/
                (np.sum(t_2dim ** 2, axis=1) - np.sum(t_2dim, axis=1)**2 / num))

            speed_map[:, X]=slope
        
        speed_map=speed_map*-1
        OCDS_volume[location] = speed_map
        t_end = time.time()
        print ("Process time for location-" + str(location) + ": " + str(t_end - t_start2) + "s")
        #break
#OCDS_volume is data to be saved as numpy file.                          
    imsave(root + '_decay speed' +str([num_frame])+ str(real_time_window_value_in_sec) + '2_4Kernelsum_abs_log1[204.8]_6[1228.8].tif'
           ,OCDS_volume.astype(dtype='f4')) #tiff file
    np.save(root+'_decay speed' +str([num_frame])+ str(real_time_window_value_in_sec) + '2_4Kernelsum_abs_log1[204.8]_6[1228.8].npy' 
            ,OCDS_volume.astype(dtype='f4')) #numpy file
    t_end = time.time()
    print ("Process time for whole locations" + ": " + str(t_end - t_start1) + "s")
#--------------------------------------------------------------------------------------------------------------------------------- 

#--------start color composition process----------------------------------------------------------------------------------------
if flag_color_composition == 0:
    sys.exit()
if flag_color_composition == 1:
        if len(np.shape(OCDS_volume)) == 2:
            flag_volume = False
            (bPerX, ptPerZ) = np.shape(OCDS_volume)
            print( "(bPerX, ptPerZ) = " + str((bPerX, ptPerZ)))
        elif len(np.shape(OCDS_volume)) == 3:
            flag_volume = True
            (bPerC, ptPerA, aPerB) = np.shape(OCDS_volume)
            print( "(bPerC, ptPerA, aPerB) = " + str((bPerC, ptPerA, aPerB)))
        else: 
            print( "Input is wrong.")
            sys.exit()
            
        with tifffile.TiffFile(path_input_array2[j]) as imfile:
            input_array2 = imfile.asarray() #Log intensity: (y, z, x)
            #####
        (bPerC, ptPerA, aPerB) = np.shape(input_array2)
        print ("(bPerC, ptPerA, aPerB) = " + str((bPerC, ptPerA, aPerB)))
        
        
        #4trial...............................................
        min_dynamic= 0#Define the min of the dynamic map
        max_dynamic = 0.0006#Define the max of the dynamic map
                    
        if flag_volume == False: #If input is 2D  
            Hue=OCDS_volume                
            # Hue[Hue!=Hue] = max_dynamic
            temp = np.clip(Hue, min_dynamic, max_dynamic)
            H = ((temp - min_dynamic) / (max_dynamic - min_dynamic) *120) #120 to make the  maximum color as geren and max SV= 20 and min=0
            
            sat=np.mean((input_array2), axis=0)
            # sat[sat!=sat] = np.nanmax(sat)
            temp = np.clip(sat, np.nanmin(sat), np.nanmax(sat))
            V = ((temp - np.nanmin(sat)) / (np.nanmax(sat) - np.nanmin(sat)) * 1)
            
            S=np.ones((pixPerA, aPerB))      #np.ones((402,512)) for taxol treated image size is 402*512
            
                                                                                                                                    
                            
            R, G, B = hsv_to_rgb(H,S, V)
            
            temp = np.concatenate([[R], [G], [B]], axis = 0)
            # temp = np.concatenate([[freq_p_norm], [fwhm_norm], [STD_norm]], axis = 0)
            rgb = np.rollaxis(temp, 0, len(np.shape(R))+1).astype('uint8')
            # hsv = np.rollaxis(temp, 0, len(np.shape(freq_p))+1).astype('f4')
            temp = np.concatenate([[H], [S], [V]], axis = 0)
            hsv = np.rollaxis(temp, 0, len(np.shape(H))+1).astype('uint8')
            
            root = os.path.splitext(path_input_array1[j])[0]
            print ("root = {}".format(root))
                
            tifffile.imsave(root + 'k2_4abs_OCDSl_' +'1[204.8]_6[1228.8]_hsv_min' + str(min_dynamic) + '-max' + str(max_dynamic) + '_volume.tif',hsv.astype('uint8'), photometric='rgb',compress=6)
            tifffile.imsave(root + 'k2_4abs_OCDSl_' +'1[204.8]_6[1228.8]_rgb_min' + str(min_dynamic) + '-max' + str(max_dynamic) + '_volume.tif',rgb.astype('uint8'), photometric='rgb',compress=6)
                
        else: #flag_volume = True
            B_scans_used_for_LIV = [32]#[4,8,32,64]
            max_time_window=511 #set the b scan number at the maximum of the time window -1
            
            ### Process for each numbers of frames
            for k in range(len(B_scans_used_for_LIV)):
                num_frame = B_scans_used_for_LIV[k]
                num_slow_scan_location = bPerC / num_frame
                num_slow_scan_location_per_unit = (max_time_window +1) / num_frame
                num_unit = bPerC / (max_time_window +1)
                
                ###Process volume (for each slow scan location)
                hsv_volume = np.zeros((num_slow_scan_location, ptPerA, aPerB, 3)) # Initialize array (y, z, x, color channel)
                rgb_volume = np.zeros((num_slow_scan_location, ptPerA, aPerB, 3)) # Initialize array (y, z, x, color channel)
                for u in range(num_unit):
                    #print "u = " + str(u)
                    for l in range(num_slow_scan_location_per_unit):
                        location_id = l + (u * num_slow_scan_location_per_unit)
                        bscan_id =np.arange (0, (max_time_window+1), (((max_time_window+1) / num_frame))) +l + (u * (max_time_window +1))
                        #print "bscan_id = " + str(bscan_id)
                        
                     
                        Hue=OCDS_volume[location_id]
                        # Hue[Hue!=Hue] = max_dynamic
                        temp = np.clip(Hue, min_dynamic, max_dynamic)
                        H = ((temp - min_dynamic) / (max_dynamic - min_dynamic) *120) #120 to make the  maximum color as geren and max SV= 20 and min=0
                        
                        sat=np.mean((input_array2[bscan_id]), axis=0)
                        # sat[sat!=sat] = np.nanmax(sat)
                        temp = np.clip(sat, np.nanmin(sat), np.nanmax(sat))
                        V = ((temp - np.nanmin(sat)) / (np.nanmax(sat) - np.nanmin(sat)) * 1)
      
                        
                        S=np.ones((pixPerA, aPerB))      #np.ones((402,512)) for taxol treated image size is 402*512
                            
                                                                                                                                                
                                        
                        R, G, B = hsv_to_rgb(H,S, V)
                        
                        temp = np.concatenate([[R], [G], [B]], axis = 0)
                        # temp = np.concatenate([[freq_p_norm], [fwhm_norm], [STD_norm]], axis = 0)
                        rgb = np.rollaxis(temp, 0, len(np.shape(R))+1).astype('uint8')
                        # hsv = np.rollaxis(temp, 0, len(np.shape(freq_p))+1).astype('f4')
                        temp = np.concatenate([[H], [S], [V]], axis = 0)
                        hsv = np.rollaxis(temp, 0, len(np.shape(H))+1).astype('uint8')
                        
                        hsv_volume[location_id] = hsv
                        rgb_volume[location_id] = rgb
            
            root = os.path.splitext(path_input_array1[j])[0]
            print ("root = {}".format(root))
                
            #temp = np.rollaxis(hsv_volume, 1, 4)#(y, z, x, channel) --> (y, x, channel, z)
            #hsv_volume = np.rollaxis(temp, 0, 4)#(y, x, channel, z) --> (x, channel, z, y)
            hsv_volume = np.rollaxis(hsv_volume, 3, 1)#(y, x, channel, z) --> (x, channel, z, y)
            
            #temp = np.rollaxis(rgb_volume, 1, 4)#(y, z, x, channel) --> (y, x, channel, z)
            #rgb_volume = np.rollaxis(temp, 0, 4)#(y, x, channel, z) --> (x, channel, z, y)
            rgb_volume = np.rollaxis(rgb_volume, 3, 1)#(y, x, channel, z) --> (x, channel, z, y)
            print (np.shape(rgb_volume))
            
            tifffile.imsave(root + 'k2_4abs_OCDSl_' +'1[204.8]_6[1228.8]_hsv_min' + str(min_dynamic) + '-max' + str(max_dynamic) + '_volume.tif',hsv_volume.astype('uint8'), photometric='rgb',compress=6)
            tifffile.imsave(root + 'k2_4abs_OCDSl_' +'1[204.8]_6[1228.8]_rgb_min' + str(min_dynamic) + '-max' + str(max_dynamic) + '_volume.tif',rgb_volume.astype('uint8'), photometric='rgb',compress=6)
                
   # print("--- %s seconds ---" % (time.time() - start_time))            
    
#---end of color composition process---------------------------------------------------------------------------------------
                                                        
                                                                                        
#                 
# for n in range(0,24,8):        
#         for X in range(512):#row index
#             for Z in range(402):#column # depth
#                 y =Corrcoef_volume[n:n+7][0:2,Z,X] # !!! change the fitting interval
#                 speed_map[Z,X]=(linregress(x[0:2], y).slope)# !!! change the fitting interval
#                 speed_map[Z,X]= (speed_map[Z,X])*-1
#                 OCDS_volume=np.append(OCDS_volume,speed_map)
#     OCDS_volume=np.reshape(OCDS_volume,(3,402,512))
#                             
#     imsave(root + 'decay speed' +str([B_scans_used_for_OCDS[k]])+ str(real_time_window_value_in_sec) + '_floatlog0_2.tif', OCDS_volume.astype(dtype='f4'))                
                


