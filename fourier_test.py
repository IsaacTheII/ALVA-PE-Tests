"""
write a testprogram that computes the fourier transform of a sequence of numbers containing a dummy signal and some noise
and plots the result. Use the fourier transform to find the frequencies.
"""

from numpy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

def fourier_test():
    """
    computes the fourier transform of a sequence of 200 numbers
    and plots the result. Use the fourier transform to find the frequencies.
    """
    signal_lenth = 200
    data_lenth = 5000

    # random padding array wiht random numbers between 0 and 1
    padding_y = np.random.random(data_lenth - signal_lenth) - 0.5
    padded_x = np.linspace(0, data_lenth, data_lenth)

    # no signal just random noice to compare difference between signal and noise
    y_rand = np.random.random(signal_lenth) - 0.5

    # define a function and add some noise
    y_x = np.linspace(0, 3.14*100, signal_lenth)
    y = np.sin(y_x) + (np.random.random(signal_lenth) - 0.5)
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) - 0.5

    # add the padding to the sequence
    split = np.random.randint(0, data_lenth - signal_lenth)
    y = np.concatenate((padding_y[:split], y, padding_y[split:]))
    y_rand = np.concatenate((padding_y[:split], y_rand, padding_y[split:]))

    # compute the fourier transform of the sequence
    yfft = fft(y)
    yfft_rand = fft(y_rand)

    # find the frequencies
    freq = fftfreq(len(y))
    freq_rand = fftfreq(len(y_rand))

    # plot the result
    plt.figure(figsize=(5, 6))
    plt.plot(padded_x, y_rand, color="blue", alpha=1)
    plt.plot(padded_x, y, color="red", alpha=0.9)

    plt.figure(figsize=(5, 6))
    plt.plot(freq_rand, yfft_rand, color="blue", alpha=1)
    plt.plot(freq, yfft, color="red", alpha=0.9)

    plt.show()


if __name__ == "__main__":
    fourier_test()
