Proj01_Chatbin
======

Code repository for project 01 cell count OCT.

## Asset list:
### Image conversion 
- Used code: ImageConverter.py
- Convert raw data (logarithmic amplitude of OCT signal after Fourier transform, both from IVS-200-HR and IVS-800) to .tif image stacks. 

>Note: Code will be upgraded after adding function to convert raw spectrum data to image stacks.  

### Volume distortion registration
- Used code: VolumeDistortRegistor.py
- Manually register all types of 3D data (*_3d_view.tif, *_IntImg_LIV.tif, *_IntImg_LIV_raw.tif) to flatten the surface. Five (can be edited) markers are manually selected to draw a contour of the surface, where a polynomial fitting (degree of 3, can be edited) is applied. 
- How to use: run, select a *_3d_view.tif, select 5 markers of the surface contour (XZ), then select another 5 markers to contour (YZ). 

### Temporal analysis
- Used code: \Temporal analysis\VarianceToRGB.py; \Temporal analysis\VarianceToViability.py
- Code to generate LIV_encoded image and raw LIV data from the linear intensity image stacks. 
>Note: The display range of the normalized LIV (hue channel of hsv color space) is set to [0, 1], but await to be further optimized based on the cellular apoptosis imaging results. 
- How to use:
1. run ImageConverter.py, to convert raw data (log int) to linear and log int image stacks (Data_IntImg.tif and Data_3d_view.tif). 
2. run VarianceToRGB.py, to encode temporal variance of log int as Hue, max log int (during raster period) as Value, 1 as Saturation. 
3. open Data_IntImg_LIV.tif in ImageJ, and measure the tilting angle along X (Bscan_tilt) and Y (Y_tilt)
4. run \Fiji_macro\AutoRotateMacro.ijm, manually set 'Bscan_tilt' and 'Y_tilt' from the above measurements, and process all 3 image stacks. 
5. open aligned LIV image (Data_IntImg_LIV.tif), and select the depth range for computing viability. 
6. run VarianceToViability.py, manually set 'zSlice' to be the determined depth range, and select the LIV image (Data_IntImg_LIV.tif) to start computing viability. 


### Counting 
- Used code: \Counting\PartivleAnaly.py; \Counting\2dCountingRendering.py: 
- Code to estimate the counting of homogeneously distributed particles using a sequence of B-scans, by computing the area fraction from the 2D images (see Dellese principle for more details); convert the computed 2D area fraction into 3D counting, and make plot which was designed for standard particles validation. 

