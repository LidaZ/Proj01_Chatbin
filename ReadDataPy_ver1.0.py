import numpy as np
import matplotlib.pyplot as plt
import tifffile
from configparser import ConfigParser


save_tif = False  # Set as True if save images as 3D stack .tiff file in the current folder
root = r"J:\Data\CellCountOCT\240423_IVS2000_MilkCof"  # Folder path which containing the raw Data


DataFold = root + r"\Data.bin"  # Raw data file name, usually it is Data.bin
conf = ConfigParser()
conf.read(root + r"\DataInfo.ini")
[dim_y, dim_x, dim_z] = [int(conf['Dimensions']['X']), int(conf['Dimensions']['Y']),
                         int(conf['Dimensions']['Z'])]  # Dimensions imported from DataInfo.ini

with open(DataFold, mode='rb') as file:
    data = np.fromfile(file, dtype='>f')  # Read data handler. ">": big-endian; "f", float32, 4 bytes
vol = data.reshape(dim_y, dim_x, dim_z)  # Reshape data as 3D volume
 

for index in range(1):  # Loop for each Y-dimension location
    plt.figure(0); plt.clf()  # Create clean figure
    plt.text(0,0, ['frame = ', index])  # Display Y-dimension location on figure
    plt.imshow(vol[index, :, :].T, cmap='gray', vmin=-10, vmax=70)
    # Show the B-scan, with display range of the signal intensity.
    # Raw data of this B-scan: vol[index, :, :].T
    # Standard: -20~90 dB
    plt.figure(2);  plt.clf();  plt.plot(vol[index, int(dim_x/2), :])  # Plot A-line profile located at middle of B-scan
    plt.pause(0.01)


if save_tif:
    tifffile.imwrite(root + r"\Data_img.tif", np.rollaxis(vol[:, :, :], 0,1))  # , compress=6)
