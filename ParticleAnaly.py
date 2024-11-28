# import pyDeepP2SA as p2sa
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import gc
import imagej
import scyjava
import pandas
import os
from tkinter import *
from tkinter import filedialog
import matplotlib
matplotlib.use("Qt5Agg")


# # # # # # # # # # # # # # # # #
# sys_ivs800 = True
# pix_sep = 5  # 5um/pix isotropic pix separation in IVS-2000-HR
# if sys_ivs800: pix_sep = 3.9  # 2um/pix for IVS-800
# plt.close(11); plt.figure(11, figsize=(13, 4));  plt.clf()


# tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True); stackFilePath = filedialog.askopenfilename(filetypes=[("", "*.csv")])
# analyExcel = os.path.basename(stackFilePath)
# folderPath = os.path.dirname(stackFilePath);  tk.destroy()
#
# # folderPath = r"J:\Data_2024\20240801_Jurkat02\lv-2hr\3d"
# # analyExcel = "Results" + ".csv"
# stackImg = "Data_3d_view" + ".tif"
# excelpath = folderPath + "\\" + analyExcel;  stackpath = folderPath + "\\" + stackImg
# # ### read stack size
# I = tifffile.imread(stackpath);  dim_y, dim_z, dim_x = I.shape;  del I;  gc.collect
# # spas = (dim_y*pix_sep/1000)*(dim_x*pix_sep/1000)*(dim_z*1.9/1000) / 1000 # 1000 mm^2 -> 1 mL
# spas = (dim_y*pix_sep/1000)*(dim_x*pix_sep/1000)*(200*1.9/1000) / 1000
# ###
# df = pandas.read_csv(excelpath)
# if 'mv' in excelpath: ptcolor = 'green'
# elif 'lv' in excelpath: ptcolor = 'red'
# elif 'hv' in excelpath: ptcolor = 'red'
# else: ptcolor = 'gray'
# # size_list = df['Area'][:]; dia_list = df['Feret'][:]; meanInt_list = df['Mean'][:]; circ_list = df['Round'][:]
# # min_dia_list = df['MinFeret'][:]  # only for test 'Particle by IVS800' avoid influence from multi reflection in bead
# meanInt_list = df['Mean'][:]
# if 'Mean dist. to surf. (pixel)' in df.columns:  dia_list = df['Mean dist. to surf. (pixel)'][:]
# elif 'Mean dist. to surf. ( )' in df.columns:  dia_list = df['Mean dist. to surf. ( )'][:]
# # dia_list = df['B-width'][:]  # width of outer box
#
# # if 'SD dist. to surf. (pixel)' in df.columns: circ_list = df['SD dist. to surf. (pixel)'][:]
# # elif 'SD dist. to surf. ( )' in df.columns:  circ_list = df['SD dist. to surf. ( )'][:]
# asp_ratio = (df['B-height'][:] * 1.9) / (dia_list * pix_sep)
#
# total_cnt =np.shape(df)[0]
# # # # mean intensity histogram
# plt.subplot(1,3,1)#.cla()
# meanInt = meanInt_list / 255
# sufix = '(-10~70 dB)'
# if sys_ivs800: sufix = '(-25~20 dB)'
# plt.hist(meanInt, facecolor=ptcolor, bins=45, range=[0.1, 0.5], alpha=0.35, density=True); plt.title('Mean Intensity per cell')
# plt.xlabel('Normalize intensity' + sufix); plt.ylabel('Density')
# plt.axis([0.2, 0.5, 0, 30]); plt.pause(0.01)
# # # # size histogram
# ax = plt.subplot(1,3,2)#.cla()
# diameter = dia_list * pix_sep
# plt.hist(diameter, facecolor=ptcolor, bins=45, range=[1, 40], alpha=0.35, density=True); plt.title('Mean size per cell');
# plt.xlabel('Diameter (um)');   plt.axis([5, 40, 0, 0.25]); #  ax.set_xscale('log')
# plt.pause(0.01)
# # # # circularity histogram
# plt.subplot(1,3,3)#.cla()
# plt.hist(asp_ratio, facecolor=ptcolor, bins=45, range=[0, 5], alpha=0.35, density=True); plt.title('STD of diameter per cell');
# plt.xlabel('Aspect ratio (a.u.)'); plt.axis([0.1, 3, 0, 4])
#
# # plt.text(1, 2.2, 'Cell count number is ' + f"{total_cnt/spas:.2E}" + "/mL")
# print('Cell count density is: ' + f"{total_cnt/spas:.2E}" + "/mL")
# index = np.argwhere(diameter<7); mod_diameter = np.delete(diameter, index)
# print('Mean diameter is: ' + str(np.around(np.mean(mod_diameter), 2)) + ' um')
# print('Mean normalized intensity is: ' + str(np.around(np.mean(meanInt), 3)))
# plt.pause(0.01)
# # # # # # # # # # # # # #


# # # Why it is not a good idea to use ImageJ macro in python:
# # https://github.com/imagej/pyimagej/issues/126
# # https://github.com/imagej/i2k-2022-pyimagej/blob/9e23cce2d0c5260dfe0bdeb7e98bcd27608f87df/I2K-2022-PyImageJ-Workshop.ipynb
scyjava.config.add_option('-Xmx6g')
root = r"C:\Users\lzhu\Desktop\OCT Data\Bayer\ProductY_20M"
DataId = "ProductY_20Mposition3_After_Mixing_timelapse_view.tif"
DataFold = root + '\\' + DataId

ij = imagej.init('sc.fiji:fiji:2.16.0', add_legacy=True)  # sc.fiji:fiji:2.16.0
image = ij.io().open(DataFold)
y_num = np.shape(image)[2]
# # _ = ij.py.from_java(image); ndImg = _.values  # convert imageJ2 dataset class to python xarray (the opposite is py.to_java()); then to numpy array
total_cnt = [0];  plt.figure(11); plt.clf(); #  plt.axis([0, y_num, 0, 1000])
for ind_y in range(y_num):
    miteimp = ij.py.to_imageplus(image[:, :, ind_y])
    # ij.py.sync_image(miteimp);  ij.py.show(miteimp, cmap='gray')
    # # # propagate the updated pixel values to numpy array (cause ij.py.show() is calling pyplot)

    # # # run plugin "Gaussian blur"
    ij.py.run_plugin("Gaussian Blur...", args={"sigma": 1.0, "stack": True}, imp=miteimp)
    # ij.py.sync_image(miteimp);  ij.py.show(miteimp, cmap='gray')

    # # # apply threshold
    imp = ij.py.to_imageplus(miteimp)
    ij.IJ.setRawThreshold(imp, 75, 255)
    # ij.IJ.run(imp, "Convert to Mask", "background=Dark black")  # Not necessary, only for monitoring
    # ij.py.sync_image(imp);  ij.py.show(imp, cmap='gray')

    # # # run plugin "Object count"
    ij.IJ.run(imp, "Analyze Particles...", "size=5-50 pixel circularity=0.20-1.00 exclude summarize")

    results = ij.ResultsTable.getResultsTable()   # command to deal with Result Table: https://imagej.net/ij/developer/api/ij/ij/measure/ResultsTable.html
    total_cnt = results.size()
    mean_cnt = total_cnt / (ind_y + 1)

    plt.figure(11); plt.scatter(ind_y, mean_cnt, color='black')
    plt.xlim(0, y_num);  plt.pause(0.01);  print(ind_y)

# # # # close instance
ij.dispose()  # legacy layer only activated once when initializing imagej, will be inactive when being called again
# # # # https://forum.image.sc/t/pyimagej-macro-run-error/68515

