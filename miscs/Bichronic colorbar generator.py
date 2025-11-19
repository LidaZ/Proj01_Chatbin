import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


size = 100
batchProj_val = np.zeros((size, size), 'float32')
for y in range(size):
    for x in range(size):
        batchProj_val[y,x] = (y/size) #* (x/size)
plt.figure(10); plt.clf(); plt.imshow(batchProj_val)

batchProj_varHue_raw = np.zeros((size, size), 'float32')
for y in range(size):
    for x in range(size):
        batchProj_varHue_raw[y,x] = (x/size) #* (x/size)

batchProj_varHue = np.multiply(np.clip((batchProj_varHue_raw-0) / (1-0), 0, 1), 0.6)
plt.figure(11); plt.clf(); plt.imshow(batchProj_varHue)

batchProj_sat = np.ones((size, size), 'float32')

batchProj_rgb = hsv_to_rgb(np.transpose([batchProj_varHue, batchProj_sat, batchProj_val]))
plt.figure(12); plt.clf(); plt.imshow(batchProj_rgb)