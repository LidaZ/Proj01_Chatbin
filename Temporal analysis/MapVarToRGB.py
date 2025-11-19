import math
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def num_to_rgb(val, max_val=3):
    i = (val * 255 / max_val);
    r = np.round(np.sin(0.024 * i + 0) * 127 + 128);
    g = np.round(np.sin(0.024 * i + 2) * 127 + 128);
    b = np.round(np.sin(0.024 * i + 4) * 127 + 128);
    return (r,g,b)


if __name__ == "__main__":
    data = np.tile(np.arange(10), (5, 1))
    data_norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max(), clip=True)
    cmap = cm.rainbow
    # mapper = cm.ScalarMappable(norm=data_norm, cmap=cmap)
    plt.figure(24); plt.clf(); plt.imshow(data, cmap=cmap)

    tmp = num_to_rgb(data)  # (ch, (data.y, data. x))

