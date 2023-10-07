import numpy as np
import matplotlib.pyplot as plt


def load_psnr_data():
    arr1 = np.fromfile("./np_arr/psnr_1.dat")
    arr2 = np.fromfile("./np_arr/psnr_7.dat")
    return arr1, arr2


def load_ssi_data():
    arr1 = np.fromfile("./np_arr/ssi_1.dat")
    arr2 = np.fromfile("./np_arr/ssi_7.dat")
    return arr1, arr2


def plot_psnr():
    x = np.linspace(1, 60, 60)
    arr1, arr2 = load_psnr_data()
    plt.figure(figsize=(8, 6))
    plt.plot(x, arr1, label='pool layer 1', color='blue')
    plt.plot(x, arr2, label='pool layer 7', color='red')
    plt.xlabel('Trainings Runde')
    plt.ylabel('dB')
    plt.legend()
    plt.title('PSNR')
    plt.grid(True)
    plt.show()


def plot_ssi():
    x = np.linspace(1, 60, 60)
    arr1, arr2 = load_ssi_data()
    plt.figure(figsize=(8, 6))
    plt.plot(x, arr1, label='pool layer 1', color='blue')
    plt.plot(x, arr2, label='pool layer 7', color='red')
    plt.xlabel('Trainings Runde')
    plt.ylabel('dB')
    plt.legend()
    plt.title('SSI')
    plt.grid(True)
    plt.show()


def run():
    plot_psnr()
    plot_ssi()
