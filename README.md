Proj01_Chatbin
======

Code repository for project 01 cell count OCT.

## Asset list:
### ImageConversion 
- Used code: ImageConverter.py
- Convert raw data (logarithmic amplitude of OCT signal after Fourier transform, both from IVS-200-HR and IVS-800) to .tif image stacks. 

>Note: Code will be upgraded after adding function to convert raw spectrum data to image stacks.  

### Temporal analysis
- Used code: VarianceToRGB.py
- Code to generate LIV_encoded image and raw LIV data from the linear intensity image stacks. 
>Note: The display range of the normalized LIV (hue channel of hsv color space) is set to [0, 1], but await to be further optimized based on the cellular apoptosis imaging results. 

### 2D Counting 
- Used code: PartivleAnaly.py; 2dCountingRendering.py: 
- Code to estimate the counting of homogeneously distributed particles using a sequence of B-scans, by computing the area fraction from the 2D images (see Dellese principle for more details); convert the computed 2D area fraction into 3D counting, and make plot which was designed for standard particles validation. 

