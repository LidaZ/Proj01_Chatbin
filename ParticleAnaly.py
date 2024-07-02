# import pyDeepP2SA as p2sa
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import gc
import imagej
import pandas
import matplotlib
matplotlib.use("Qt5Agg")

# root = r"J:\Bayer Data\ProductY_5M"
# DataId = "bscan_ivs2k_5m.tif"
# DataFold = root + '\\' + DataId
# I = plt.imread(DataFold)
### failed demo to use pyDeepP2SA (SegmentAnythingModel from MetaAI, Facebook Inc.) for OCT particle segment
### https://pypi.org/project/pyDeepP2SA/
### https://github.com/facebookresearch/segment-anything#model-checkpoints
# image = np.swapaxes(np.swapaxes(np.array([I, I, I]), 0, 1), 1, 2)
# sam_checkpoint = r"C:\Users\lzhu\PycharmProjects\CellCountProj\sam_vit_h_4b8939.pth"
#
# masks = p2sa.generate_masks(image, sam_checkpoint, points_per_side=20, pred_iou_thresh=0.75,
#                       stability_score_thresh=0.75, crop_n_layers=1, crop_n_points_downscale_factor=2,
#                       min_mask_region_area=5)
# # csv_directory = root + '\\' + "masks.csv"
# # p2sa.save_masks_to_csv(masks, csv_directory, 1)
# # p2sa.generate_masks(image, masks)
# p2sa.plot_diameters(image, masks, 5, 0.7, 1)
###failed demo to use imageJ macro to call Biovoxxxel 3D box > object inspector plugin
# macrotest = """
# run("Duplicate...", " ");
# setThreshold(50, 255, "raw");
# setThreshold(50, 255, "raw");
# //setThreshold(50, 255);
# run("Convert to Mask");
# run("Watershed");
# run("Object Inspector (2D/3D)", "primary_imageplus=bscan_ivs2k_5m-1.tif secondary_imageplus=bscan_ivs2k_5m.tif original_1_title=None original_2_title=None primary_volume_range=7-80 primary_mmer_range=0.70-1.00 secondary_volume_range=0-5000 secondary_mmer_range=0.00-1.00 exclude_primary_objects_on_edges=false pad_stack_tops=false display_results_tables=true display_analyzed_label_maps=false show_count_map=true");
# selectImage("bscan_ivs2k_5m-1.tif");
# close();
# """
# ij = imagej.init('sc.fiji:fiji')
# result = ij.py.run_macro(macrotest)
####################
# plt.figure(10); plt.clf(); plt.imshow(I, cmap='gray'); plt.pause(0.01)

sys_ivs800 = True
pix_sep = 5  # 5um/pix isotropic pix separation in IVS-2000-HR
if sys_ivs800: pix_sep = 2  # 2um/pix for IVS-800
plt.close(11); plt.figure(11, figsize=(13, 4));  plt.clf()


folderPath = r"F:\Data_2024\20240626_jurkat\mv-3hr\3DAnalysis"
analyExcel = "Statistics for Data_3d_view" + ".csv"
stackImg = "Data_3d_view" + ".tif"
excelpath = folderPath + "\\" + analyExcel;  stackpath = folderPath + "\\" + stackImg
# ### read stack size
I = tifffile.imread(stackpath);  dim_y, dim_z, dim_x = I.shape;  del I;  gc.collect
spas = (dim_y*pix_sep/1000)*(dim_x*pix_sep/1000)*(dim_z*1.9/1000) / 1000 # 1000 mm^2 -> 1 mL
###
df = pandas.read_csv(excelpath)
if 'mv' in excelpath: ptcolor = 'green'
elif 'lv' in excelpath: ptcolor = 'red'
elif 'hv' in excelpath: ptcolor = 'gray'
else: ptcolor = 'gray'
# size_list = df['Area'][:]; dia_list = df['Feret'][:]; meanInt_list = df['Mean'][:]; circ_list = df['Round'][:]
# min_dia_list = df['MinFeret'][:]  # only for test 'Particle by IVS800' avoid influence from multi reflection in bead
meanInt_list = df['Mean'][:]
if 'Mean dist. to surf. (pixel)' in df.columns:  dia_list = df['Mean dist. to surf. (pixel)'][:]
elif 'Mean dist. to surf. ( )' in df.columns:  dia_list = df['Mean dist. to surf. ( )'][:]
if 'SD dist. to surf. (pixel)' in df.columns: circ_list = df['SD dist. to surf. (pixel)'][:]
elif 'SD dist. to surf. ( )' in df.columns:  circ_list = df['SD dist. to surf. ( )'][:]
total_cnt =np.shape(df)[0]
# # # mean intensity histogram
plt.subplot(1,3,1)#.cla()
meanInt = meanInt_list / 255
sufix = '(-10~70 dB)'
if sys_ivs800: sufix = '(-25~20 dB)'
plt.hist(meanInt, facecolor=ptcolor, bins=50, range=[0.1, 0.5], alpha=0.35, density=True); plt.title('Mean Intensity per cell')
plt.xlabel('Normalize intensity' + sufix); plt.ylabel('Density')
plt.axis([0.1, 0.45, 0, 30]); plt.pause(0.01)
# # # size histogram
plt.subplot(1,3,2)#.cla()
diameter = dia_list * pix_sep
plt.hist(diameter, facecolor=ptcolor, bins=50, range=[1, 40], alpha=0.35, density=True); plt.title('Mean size per cell');
plt.xlabel('Diameter (um)'); plt.axis([0, 40, 0, 0.35])
plt.pause(0.01)
# # # circularity histogram
plt.subplot(1,3,3)#.cla()
plt.hist(circ_list, facecolor=ptcolor, bins=50, range=[0, 5], alpha=0.35, density=True); plt.title('STD of diameter per cell');
plt.xlabel('Regularity (A.u.)'); plt.axis([0, 5, 0, 1.75])

# plt.text(1, 2.2, 'Cell count number is ' + f"{total_cnt/spas:.2E}" + "/mL")
print('Cell count number is ' + f"{total_cnt/spas:.2E}" + "/mL")
plt.pause(0.01)

