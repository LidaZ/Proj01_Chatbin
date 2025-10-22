import sys
import numpy as np
import tifffile
import os
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
# from mpl_point_clicker import clicker
from matplotlib.backend_bases import MouseButton
import matplotlib.patches as patches
import matplotlib
matplotlib.use("Qt5Agg")


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
                try:
                    pick_point = (int(event.xdata), int(event.ydata))
                except TypeError:
                    raise ValueError("Select within the frame")
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
            raise ValueError("Selection status not reset.")

    def return_coord_in_frame(self, ax, fig, image, title_text):
        """Display the image and allow contour selection."""
        self.pick_stack = []
        self.select_not_done = True
        self.select_count = 0

        ax.clear()
        ax.imshow(image, cmap="gray")
        ax.title.set_text(title_text)

        cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)  # connect the event signal (press, drag, release) to the callback functions
        while self.select_not_done:
            plt.pause(0.5)
        fig.canvas.mpl_disconnect(cid_press)  # disconnect event handler, using the same cid
        print("Selected contour coordinates:", self.pick_stack)
        return self.pick_stack

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
    volume = tifffile.imread(volume_path)
    try:
        registered_volume = picker.fast_roll_along_z(volume, offSetMap)
        tifffile.imwrite(volume_path, registered_volume)
        print(f"Registration complete, ID: " + str(volume_path))
    except ValueError:
        raise(ValueError(f"Shape not match, ID: " + str(volume_path)))


# # # =========================
# # # Usage Example
# # # =========================

# # # Initialize Tkinter for file selection
tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True)
stack_file_path = filedialog.askopenfilename(filetypes=[("", "*view.tif")])
tk.destroy()
# # # Load image
print('Loading fileID: ' + stack_file_path)
raw_data = tifffile.imread(stack_file_path)
dim_y, dim_z, dim_x = raw_data.shape[0:3]
# # # Initialize display window and picker
picker = PickContour(contour_points=5)
fig1 = plt.figure(10, figsize=(9, 9));  plt.clf()
ax1 = fig1.subplot_mosaic("ab")


# # # Select contour in X-Z plane; then polynomial fitting to obtain the functions of contour and draw on image.
x_plane_contour_coords = picker.return_coord_in_frame(
    ax1["a"], fig1, raw_data[round(dim_y/2), ...], "5 Clicks to contour \n X-Z distortion")
x_xz_fitted, y_xz_fitted, xz_poly = picker.polyContour(x_plane_contour_coords, dim_y, ax1["a"])
# # # Select contour in Y-Z plane
y_plane_contour_coords = picker.return_coord_in_frame(
    ax1["b"], fig1, raw_data[..., round(dim_x/2)].T, "5 Clicks to contour \n Y-Z distortion")
x_yz_fitted, y_yz_fitted, yz_poly = picker.polyContour(y_plane_contour_coords, dim_x, ax1["b"])


# # # Make offset map for registration
xIndex = np.linspace(0, dim_x-1, dim_x); xMap = xz_poly(xIndex)
yIndex = np.linspace(0, dim_y-1, dim_y); yMap = yz_poly(yIndex)
xMap2d = np.tile(xMap - xMap[0], (dim_y, 1))
yMap2d = np.tile((yMap - yMap[0]), (dim_x, 1)).T
offSetMap = np.trunc(xMap2d + yMap2d).astype(np.int16) * -1

# # # # =========================
# # # # Example: register raw_data volume by using offsetMap
# RegisterVol = picker.fast_roll_along_z(raw_data, offSetMap)
# # # # # # Check if offset map coordinates with ortho-slices
# # # # ax1["a"].clear(); ax1["a"].imshow(RegisterVol[124, ...], cmap='gray'); ax1["a"].title.set_text("After X-Z registration")
# # # # ax1["b"].clear(); ax1["b"].imshow(RegisterVol[..., 245].T, cmap='gray'); ax1["b"].title.set_text("After Y-Z registration")
# # # # =========================

del raw_data

# # # overwrite the original .tif files with registered volume
DataId = os.path.basename(stack_file_path)
root = os.path.dirname(stack_file_path)

volumeRegistor((root + '/' + DataId[:-12] + '_IntImg_LIV_raw.tif'), offSetMap, picker)
volumeRegistor((root + '/' + DataId[:-12] + '_IntImg_LIV.tif'), offSetMap, picker)
volumeRegistor((root + '/' + DataId[:-12] + '_3d_view.tif'), offSetMap, picker)