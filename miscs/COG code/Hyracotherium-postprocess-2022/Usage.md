Usage of Hyracotherium-postprocess-2022
=======================================

About this program
------------------
This program is for post-processing OCT data with LIV, dynamic OCT algorithms.

### Available algorithms
- LIV
- OCDS
- damping (not damping factor (DF) of VLIV algorithm)

### Functions to compute each algorithm
If you don't need to compute some of them, please comment out the corresponding computation functions, and also comment out colorization functions and data save functions below the computation function.
- LIV : doct.computeLivAndIntensity()
- OCDS : doct.computeOcds()
- damping : doct.computeOcds() and computeDamp = True

### Preparation steps for processing
- Imput your data in "inputFilePath = [ __ ]"
- Set file type "fileType" and signal scale "signalScale".
    - self.frameType =
        - tiff: COG intensity tiff file (32-bit float, big endian)
        - npy : NumPy binary.
        - 3dv : COG intensity 3dv file (32-bit float, big endian)
        - NtuBin : Binary file (little endian) measured by NTC/CGU C++ measurement program/ CGU LabVIEW measurement program.
        - CguBin : Binary file (big endian) measured by CGU LabVIEW measurement program.
    - signalScale = 'linear' or 'dB'
- Set necessity of motion correction "setcorrmethord".
	- None : don't apply motion correction.
	- testMethod : apply motion correction with method of Morishita2023BOE.
- (case of setcorrmethord = "testMethod") Set starting depth to register "register_start_depth".
- Set kernel averaging pixel size for blur image "blurnumber" (kernel averaging is applied in colorizing.)
- Set the measurement parameters "rastParam"
	- aPerFram : num of A-lines per one B-scan frame
	- frameRepeats : repeating number of frames
	- bscanPerBlock : num of B-scans per one block
	- blockRepeats : repeating number of blocks
	- blockPerVolume : num of blocks per one volume
- Set frame separation time "frameSeparationTime", pixel bite "pixByte", sampling point in one A-line "pixPerA".
- Set dynamic ranges for pseudo color images "inputRanges = [(0, 30), (35, 60)]".
	- First ( , ) should be the dynamic range of LIV (OCDS), which is used as hue, second ( , ) should be the dynamic range of dB-scale intensity, which is used as saturation.
	
	





