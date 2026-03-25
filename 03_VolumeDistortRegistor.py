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
fig1 = plt.figure(10, figsize=(9, 9));  plt.clf()
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