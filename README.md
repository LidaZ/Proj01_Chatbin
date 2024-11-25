Proj01_Chatbin
======

Code repository for project 01 cell count OCT.

## Asset list:
### ImageConverte_verx.py: 
- Code to convert raw data (amplitude of OCT signal after Fourier transform, both from IVS-200-HR and IVS-800) to .tif image stacks. 

>Note: Code name will be changed as "ImageConverter_ver2.0.py" after adding function to convert raw spectrum data to image stacks.  

### VarianceToRGB.py
- Code to generate LIV_encoded image and raw LIV data from the linear intensity image stacks. 
>Note: The display range is set to [0, 1], but await to be further optimized based on the cellular apoptosis imaging results. 

### PartivleAnaly.py: 
- Code to make plot from the result of "particle analysis" built-in function of Fiji.

