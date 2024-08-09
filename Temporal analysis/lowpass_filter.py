from scipy.signal import butter,filtfilt
import numpy as np
import gc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


def butter_lowpass_filter(data, cutoff, fs, order, axis):
    normal_cutoff = cutoff / (0.5 * fs)
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=axis)
    return y


def sin_wave(A, f, fs, phi, t):
    Ts = 1/fs
    n = np.arange(t / Ts)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y


if __name__ == "__main__":
    wav1 = sin_wave(2, 10, 10, 0, 100)
    wav2 = sin_wave(2, 30, 10, 0, 100)
    data_2d = np.tile(wav1, (10, 1))
    data_2d[5:10] = np.tile(wav2, (5, 1))
    data_lp = butter_lowpass_filter(data_2d, 0.1, 5, 2)
    plt.figure(22); plt.clf(); plt.plot(data_2d[5, :]); plt.plot(data_lp[6, :])
    gc.collect()