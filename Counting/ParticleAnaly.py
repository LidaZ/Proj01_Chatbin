# import pyDeepP2SA as p2sa
import numpy as np
import matplotlib.pyplot as plt
# import PIL.Image
# import imageio
import tifffile
import gc
# import imagej
# import time
# import scyjava
# import pandas
import sys
import os
from tkinter import *
from tkinter import filedialog
from cellpose import denoise#, utils, io
import matplotlib
matplotlib.use("Qt5Agg")



# # # Why it is not a good idea to use ImageJ macro in python:
# # https://github.com/imagej/pyimagej/issues/126
# # https://github.com/imagej/i2k-2022-pyimagej/blob/9e23cce2d0c5260dfe0bdeb7e98bcd27608f87df/I2K-2022-PyImageJ-Workshop.ipynb
# scyjava.config.add_option('-Xmx6g')
# root = r"C:\Users\lzhu\Desktop\OCT Data\20241218_10umBeads_4Density_IVS2000\Den0.02a\Flow0.6"
# DataId = "Data_3d_view.tif"
# DataFold = root + '\\' + DataId
tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True); DataFold = filedialog.askopenfilename(filetypes=[("", "*.tif")])
tk.destroy();  #DataId = os.path.basename(DataFold);   root = os.path.dirname(DataFold)

# ij = imagej.init('sc.fiji:fiji:2.16.0', add_legacy=True)  # sc.fiji:fiji:2.16.0
# rawData = ij.io().open(DataFold)
rawData = np.transpose(tifffile.imread(DataFold), (2, 1, 0))
x_num, z_num, y_num = np.shape(rawData)  # may be different for IVS-800 data form

# # _ = ij.py.from_java(image); ndImg = _.values  # convert imageJ2 dataset class to python xarray (the opposite is py.to_java()); then to numpy array
total_cnt = [0];  gifImg = [];  framList = [0];  areaFraction_list = [0]
# fig_frac, ax_frac = plt.subplots(1, 1)  #  plt.axis([0, y_num, 0, 1000])
fig, ax = plt.subplot_mosaic("ABC;ABC;ABC;...;DDD;DDD")  # ("AAD;BBD;CCD")
fig.set_dpi(150)
ax['A'].title.set_text('Before denoising'); ax['B'].title.set_text('After denoising'); ax['C'].title.set_text('Masking'); ax['D'].title.set_text('Mean area fraction')
# io.logger_setup()
model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")
intThreshold = 0.55  # 135
imgs_dn = [[0], [0]]
# start = time.time()
for ind_y in range(y_num):  # y_num
    img = rawData[:, :, ind_y]
    try:  ax_a.set_data(img)
    except NameError:  ax_a = ax['A'].imshow(img, cmap='gray')
    # miteimp = ij.py.to_imageplus(image[:, :, ind_y])  # convert python xarray to imageJ2 dataset
    # img = ij.py.from_java(image[:, :, ind_y])
    # ij.py.sync_image(miteimp);  ij.py.show(miteimp, cmap='gray')
    # # # propagate the updated pixel values to numpy array (cause ij.py.show() is calling pyplot)

    # # # # run plugin "Gaussian blur".
    # # # # Update: no longer needed after implementing Cellpose v3, since there is already Gaussian blurring applied.
    # gaussBlur = 1.5
    # ij.py.run_plugin("Gaussian Blur...", args={"sigma": gaussBlur, "stack": True}, imp=miteimp)
    # ij.py.sync_image(miteimp);  ij.py.show(miteimp, cmap='gray')

    # # # apply Cellpose v3 for denoising
    try:
        masks, flows, styles, imgs_dn = model.eval(img, diameter=None, channels=[0, 0]) #, niter=200000)

        # # # imgs_dn is the normalized denoised image; diameter=5 seems better than 0/None and dia=7 (not sure if this setting works)
        try:  ax_b.set_data(imgs_dn)
        except NameError:  ax_b = ax['B'].imshow(imgs_dn, cmap='gray')
        # # # segmentation, model may need re-train for segmentation, since the model was trained by resized images where mean diameter = 30 pix,
        # # # One can resize the images so that "10 um = 30 pix", but in cell count OCT it may be risky to resize 3X due to resolution is already low.
        # outlines = utils.outlines_list(masks)
        # for o in outlines:  ax['A'].plot(o[:, 0], o[:, 1], color=[1, 1, 0])

        # # # # threshold to obtain area fraction from frame
        imgs_dnThresh = (imgs_dn > intThreshold) * imgs_dn
        try:  ax_c.set_data(imgs_dnThresh)
        except NameError:  ax_c = ax['C'].imshow(imgs_dnThresh, cmap='gray')
        areaFraction = np.count_nonzero(imgs_dnThresh) / np.size(imgs_dn)
        areaFraction_list.append(areaFraction)
        meanAreaFraction = np.mean(areaFraction_list)
        # imp = ij.py.to_imageplus(miteimp)
        # ij.IJ.setRawThreshold(imp, intThreshold, 255)  # better to make it auto-thresholding
        # # ij.IJ.run(imp, "Convert to Mask", "background=Dark black")  # Not necessary, only for monitoring
        # # ij.py.sync_image(imp);  ij.py.show(imp, cmap='gray')

        # # # # run plugin "Object count"
        # ij.IJ.run(imp, "Analyze Particles...", "size=10-200 pixel circularity=0.00-1.00 exclude summarize")
        #
        # results = ij.ResultsTable.getResultsTable()   # command to deal with Result Table: https://imagej.net/ij/developer/api/ij/ij/measure/ResultsTable.html
        # total_cnt = results.size()
        # mean_cnt = total_cnt / (ind_y + 1)
        #
        if ind_y%(y_num//100) == 0:  ax['D'].scatter(ind_y, meanAreaFraction, color='#6ea6db', marker='o', s=7)#, facecolors='none')
        # # # plt.xlim(0, y_num);  plt.pause(0.01);  print(ind_y)
        # # # fig.savefig(r"C:\Users\lzhu\Desktop\tmp.png")
        # # # im = imageio.v2.imread(r"C:\Users\lzhu\Desktop\tmp.png")
        # # # gifImg.append(im)
    except IndexError:
        pass  # avoiding index error of torch in dynamics.py

    sys.stdout.write('\r')
    j = (ind_y + 1) / y_num
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j) + ' on batch processing')
    plt.pause(0.01)

# # # # # close instance
# ij.dispose()  # legacy layer only activated once when initializing imagej, will be inactive when being called again
# # # # # https://forum.image.sc/t/pyimagej-macro-run-error/68515
# imageio.mimsave(r"C:\Users\lzhu\Desktop\animated_plot.gif", gifImg, format='Gif', fps=30, loop=0)
# end = time.time(); print(end - start)

print('Mean area fraction is: ', meanAreaFraction)
# record of pixels per slice of a 10-um particle: 4+21+41+63+73+90+92(> mean diameter = 5.5 pix)+89+83+67+53+26+6 = 708 pixel / particle
