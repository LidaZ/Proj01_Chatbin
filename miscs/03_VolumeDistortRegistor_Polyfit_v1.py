import sys
import numpy as np
import tifffile
import os
import glob
from tkinter import *
from tkinter import filedialog
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
# from mpl_point_clicker import clicker
from matplotlib.backend_bases import MouseButton
import matplotlib.patches as patches


""" 
Select five points to draw contours of distortion in X-Z and Y-Z planes. 

Prompt: 
我目前代码实现的功能是通过手动选择物体表面轮廓5点+多项式拟合的方法，将物体的三维图像沿着表面轮廓“拉平”（或者叫registration）。
具体实现方式如下：读取一个三维tiff图，显示两幅过中心的XZ、YZ截面图（对应代码中的`raw_data[round(dim_y/2), ...]`和`raw_data[..., round(dim_x/2)].T`），用户分别在XZ图和YZ图上沿着物体表面手动选择5个点，代码对选择的点各自进行多项式拟合，并插值到3D空间，得到物体表面轮廓曲面，最后沿着这个曲面对整个三维图像进行“拉平”。
目前不足的地方在于，表面轮廓的拉平精度极度依赖用户选择点是否准确、多项式拟合是否符合实际物体表面轮廓，无法做到根据图像中的实际轮廓信息自动计算出轮廓曲线、曲面。我有考虑过引入机器视觉模块来自动判断轮廓曲线，但是实际应用时，由于图像内容的复杂性，我期望拉平的轮廓曲线的周围（上下方向）常有其他伪影存在，且表面轮廓所在的Z轴位置常常不固定，所以我担心无法做到自动识别哪一段Z轴区域内才适用于自动轮廓识别。
目前我的想法是，同样还是显示两幅过中心的XZ、YZ截面图，用户还是可以在两幅图中手动选择若干个（例如5个）表面轮廓的点。但是不再使用多项式拟合这种低精度的方法去估计表面轮廓曲线，而是使用其他更robust的自动表面轮廓识别算法（细节在最后一段进行补充)，只是其适用范围不再是整幅XZ图和YZ图，而是在用户选择点周围的一小段Z轴区域内。计算出XZ、YZ截面图中的表面轮廓曲线后，先尝试对这两幅图进行表面拉平，并更新图像让用户检验拉平效果是否合格（如果不合格的话可以在代码里微调“用户选择点周围的一小段Z轴区域”的区域范围，比如说上下50pixel > 10pixel）。如果用户觉得拉平效果合格、并点击确认后，算法自动估算出整个三维图中的合适的“用户选择点周围的一小段Z轴区域”，并在该范围内提取表面轮廓曲面，最后对整个三维图进行拉平操作。
有一点需要注意的是，数据集中常见的物体表面轮廓，通常并非是一小段范围内的最高强度点，因为物体通常为强散射+低轴向分辨率，意味着物体表面轮廓处的特征更多是强度从低到高的突变，且表面轮廓的边界处通常稍显模糊、不会特别锋利，即对比度不会很理想；同时，也会有其表面轮廓的上下部分同时存在其他强散射的分层结构。关键是这些“非表面轮廓”的分层结构与目标的表面轮廓一样，有几乎相同的轮廓线（可以理解为只加了不同的Z方向的offset）。所以有时候当样品的目标表面轮廓层的对比度不清晰时，用户可以手动选择那些对比度更高的“非表面轮廓”层，来达到同样的三维图拉平效果，***并且当检测出多个边缘曲线时，倾向于选择更贴近“用户选择点”一侧的轮廓线***）。所以我猜测，简单的“平滑+峰值检测”可能无法满足这个情况，请仔细论证并使用最适合、robust的轮廓检测算法。
我想继续新增一个需求，在"Adjust parameter"窗口，当用户点击Cancel之后，结束当前运行的程序
还有一个小问题，在执行xz_anchors = picker.collect()yz_anchors = picker.collect()的时候，显示的图像由于是256(X) x 800 (Z)尺寸的（这里只是举例，大多数情况都是Z轴像素多余X或Y轴），在显示时沿Z方向填满后，X或Y方向显得很小，使用户很难肉眼观察并进行手动标记。能否让图像显示时强行沿X或Y方向填满，Z方向若有溢出的部分也无所谓，我可以稍后通过调整plt.subplot(figsize)来优化显示效果。
现在我想减少fig, axes_all = plt.subplots(2, 2, figsize=(8, 15), num=10)内四个子窗口周围的留白部分来继续增大图像显示面积。
"""

class PickContour:
    def __init__(self, contour_points=5):
        """Initialize contour selection."""
        self.contour_points = contour_points
        self.pick_stack = []
        self.select_not_done = True
        self.select_count = 0
        self.polyDegree = 3

    def on_press(self, event):
        """Handle mouse click events for point selection."""
        if self.select_not_done:
            if event.button is MouseButton.LEFT:
                if event.xdata is None or event.ydata is None:
                    print("Select within the frame")
                    return
                pick_point = (int(event.xdata), int(event.ydata))
                print(f"Pick coordinate: {int(event.xdata)} {int(event.ydata)};  "
                    f"{self.select_count+1}/{self.contour_points} clicks")

                self.pick_stack.append(pick_point)
                self.select_count += 1
                if self.select_count >= self.contour_points:
                    self.select_not_done = False

            elif event.button is MouseButton.RIGHT:
                # Right-click to cancel the last selected point
                if self.select_count > 0:
                    removed_point = self.pick_stack.pop()
                    self.select_count -= 1
                    print(f"Cancelled coordinate: {removed_point};  "
                          f"Remaining: {self.select_count}/{self.contour_points} clicks")
                else:   print("No points to cancel.")

        else:
            print("Selection done. Waiting for confirmation.")

    def on_key(self, event):
        """Handle key press events for confirmation or reset."""
        if event.key == 'enter':
            if not self.select_not_done:
                self.confirmed = True
                print("Confirmed selection.")
        elif event.key == 'r':
            self.reset_selection = True
            print("Resetting selection...")

    def return_coord_in_frame(self, ax, fig, image, title_text):
        """Display the image and allow contour selection with preview and confirmation."""
        while True:
            self.pick_stack = []
            self.select_not_done = True
            self.select_count = 0
            self.confirmed = False
            self.reset_selection = False

            ax.clear()
            ax.imshow(image, cmap="gray")
            ax.title.set_text(title_text + "\n(Left-click: select, Right-click: undo)")
            fig.canvas.draw()

            cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
            cid_key = fig.canvas.mpl_connect('key_press_event', self.on_key)

            while self.select_not_done and not self.reset_selection:
                plt.pause(0.1)

            if not self.reset_selection:
                # Show fitting preview
                x_smooth, y_smooth, poly = self.polyContour(self.pick_stack, image.shape[1], ax)
                ax.title.set_text(title_text + "\n(Enter: Confirm, R: Re-select)")
                fig.canvas.draw()

                while not self.confirmed and not self.reset_selection:
                    plt.pause(0.1)

            fig.canvas.mpl_disconnect(cid_press)
            fig.canvas.mpl_disconnect(cid_key)

            if self.confirmed:
                return self.pick_stack, poly

    def polyContour(self, contour_coordinates, dimension_size, ax):
        """Perform polynomial fitting on contour points."""
        x, y = zip(*contour_coordinates)
        poly_coeffs = np.polyfit(x, y, self.polyDegree)
        poly = np.poly1d(poly_coeffs)
        x_smooth = np.linspace(1, dimension_size-1, 100)
        y_smooth = poly(x_smooth)

        ax.scatter(x, y, color='red', label="Contour Points")  # Original points
        ax.plot(x_smooth, y_smooth, 'b-', label="Fitted Curve")  # Fitted curve
        plt.pause(0.01)
        return x_smooth, y_smooth, poly

    def fast_roll_along_z(self, volume, offSet_map):
        """ Efficiently rolls a 3D array A along the dim_z axis based on offSet_map. """
        dim_y, dim_z, dim_x = volume.shape[0:3]
        assert offSet_map.shape == (dim_y, dim_x), "offSetMap shape mismatch with input volume shape"

        # # # # Using numpy.roll to register data volume with estimated offsetMap
        # y_indices, x_indices = np.meshgrid(np.arange(dim_y), np.arange(dim_x), indexing='ij')
        # rolled_vol = np.empty_like(volume)
        # for y, x in zip(y_indices.ravel(), x_indices.ravel()):
        #     rolled_vol[y, :, x] = np.roll(raw_data[y, :, x], offSetMap[y, x], axis=0)  # Roll along dim_z

        # Generate a base index for the dim_z axis
        z_indices = np.arange(dim_z).reshape(1, dim_z, 1)  # Shape: (1, dim_z, 1)
        # Compute new indices with offset wrapping (modulo for circular shift)
        new_z_indices = (z_indices - offSet_map[:, None, :]) % dim_z  # Shape: (dim_y, dim_z, dim_x)
        # Use advanced indexing to reorder elements efficiently
        rolled_vol = volume[np.arange(dim_y)[:, None, None], new_z_indices, np.arange(dim_x)[None, None, :]]
        return rolled_vol


def volumeRegistor(volume_path, offSetMap, picker):
    """ Apply the estimated offSetMap to register given 3D volume. """
    volume = load_full_tiffstack(volume_path)
    try:
        registered_volume = picker.fast_roll_along_z(volume, offSetMap)
        tifffile.imwrite(volume_path, registered_volume)
        print(f"Registration complete, ID: " + str(volume_path))
    except ValueError:
        raise(ValueError(f"Shape not match, ID: " + str(volume_path)))


def load_full_tiffstack(path):
    with tifffile.TiffFile(path) as tif:
        num_pages = len(tif.pages)
        volume_full = tif.asarray(key=range(num_pages), out='memmap')
    return volume_full

# # # =========================
# # # Usage Example
# # # =========================

# # # Initialize Tkinter for file selection
tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True)
file_name_filter = "*.bin"
stack_file_path = filedialog.askopenfilename(filetypes=[("", file_name_filter)])
tk.destroy()
# # # Load image
DataId = os.path.basename(stack_file_path); root = os.path.dirname(stack_file_path)
string_DataId = DataId[:-4]
print('Loading fileID: ' + root)
raw_data = load_full_tiffstack(root + '/' + string_DataId + '_3d_view.tif')
dim_y, dim_z, dim_x = raw_data.shape[0:3]
# # # Initialize display window and picker
picker = PickContour(contour_points=5)
fig1 = plt.figure(10, figsize=(12, 12));  plt.clf()
ax1 = fig1.subplot_mosaic("ab")


# # # Select contour in X-Z plane; then polynomial fitting to obtain the functions of contour and draw on image.
x_plane_contour_coords, xz_poly = picker.return_coord_in_frame(
    ax1["a"], fig1, raw_data[round(dim_y/2), ...], "5 Clicks to contour \n X-Z distortion")

# # # Select contour in Y-Z plane
y_plane_contour_coords, yz_poly = picker.return_coord_in_frame(
    ax1["b"], fig1, raw_data[..., round(dim_x/2)].T, "5 Clicks to contour \n Y-Z distortion")

plt.close(fig1)
# # # Make offset map for registration
xIndex = np.linspace(0, dim_x-1, dim_x); xMap = xz_poly(xIndex)
yIndex = np.linspace(0, dim_y-1, dim_y); yMap = yz_poly(yIndex)
xMap2d = np.tile(xMap - xMap[0], (dim_y, 1))
yMap2d = np.tile((yMap - yMap[0]), (dim_x, 1)).T
offSetMap = np.trunc(xMap2d + yMap2d).astype(np.int16) * -1

del raw_data

# # # overwrite the original .tif files with registered volume
DataId = os.path.basename(stack_file_path)
root = os.path.dirname(stack_file_path)

# if 'view' in DataId:
volumeRegistor((root + '/' + string_DataId + '_3d_view.tif'), offSetMap, picker)

try:     volumeRegistor((root + '/' + string_DataId + '_IntImg_meanFreq.tif'), offSetMap, picker)
except FileNotFoundError:     print('Mean frequency mode is off, skip saving as image.')
try:
    volumeRegistor((root + '/' + string_DataId + '_IntImg_LIV_raw.tif'), offSetMap, picker)
    volumeRegistor((root + '/' + string_DataId + '_IntImg_LIV.tif'), offSetMap, picker)
except FileNotFoundError:     print('No LIV data found, skip saving as image.')
try:
    volumeRegistor((root + '/' + string_DataId + '_IntImg_mLIV_raw.tif'), offSetMap, picker)
    volumeRegistor((root + '/' + string_DataId + '_IntImg_mLIV.tif'), offSetMap, picker)
except FileNotFoundError:     print('No mLIV data found.')

# elif 'dbOct' in DataId:
try:
    volumeRegistor((root + '/' + string_DataId + '_IntImg_aliv.tif'), offSetMap, picker)
    volumeRegistor(glob.glob(root + '/' + string_DataId + '_IntImg_aliv_min*-max*.tif')[0], offSetMap, picker)
    volumeRegistor((root + '/' + string_DataId + '_IntImg_dbOct.tif'), offSetMap, picker)
    volumeRegistor((root + '/' + string_DataId + '_IntImg_swiftness.tif'), offSetMap, picker)
    volumeRegistor(glob.glob(root + '/' + string_DataId + '_IntImg_swiftness_min*-max*.tif')[0], offSetMap, picker)
except FileNotFoundError:     print('No aLiv or swiftness images found.')