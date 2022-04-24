import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pathlib

pi = np.pi

x = np.linspace(0, 2 * pi, num=50)
xnew = np.linspace(0, 4 * pi, num=100, endpoint=True)


def generate_data(num, plot=False, saveCSV=False, saveNPY=False, ret=False):
    """Generates plots and saves in both .npy, .csv formats multiple data arrays used later for testing GMM
     and GMR capabilities.
    Parameters
    ----------
    num : int
        Number of random array demonstrations that are generated
    plot : bool (default: False)
        Decides if you want to plot the generated data
    saveCSV : bool (default: False)
        Decides if you want to save the data as .csv format
    saveNPY : bool (default: False)
        Decides if you want to save the data as .npy format
    ret : bool (default: False)
        Decides if you want to return the generated data as numpy parameter
    Returns
    -------
    f : array, shape (num, 100)
        If the parameter 'ret' is True that it return the generated data as npy array
    """
    n = np.zeros((num, 50))
    yf = np.zeros((num, 50))
    y = np.zeros((num, 100))

    f = []
    for i in range(num):
        noise = np.random.normal(0, 0.01, 50)
        r = np.random.random_sample()
        n[i] = (1 + r / 5) * (1 + np.cos(x)) + noise + 0.09
        r = np.random.random_sample() + 0.5
        noise = np.random.normal(0, 0.01, 50)
        yf[i] = 2.5 * np.exp(-0.5 * r / x) + noise
        aux = np.concatenate((yf[i], n[i]))
        y[i] = aux
        f.append(interp1d(xnew, y[i], kind='cubic'))

    if plot:
        for i in range(num):
            plt.plot(xnew, f[i](xnew))
        plt.show()

    if saveCSV:
        file = open(str(pathlib.Path(__file__).parent.resolve()) + '/data.csv', 'w')
        writer = csv.writer(file)
        header = ['nr', 'x']
        for i in range(num):
            header.append('f' + str(i))
        writer.writerow(header)
        for j in range(len(xnew)):
            row = [str(j), str(xnew[j])]
            for i in range(num):
                row.append(str(f[i](xnew[j])))
            writer.writerow(row)
        file.close()

    if saveNPY:
        file = open(str(pathlib.Path(__file__).parent.resolve()) + '/data.npy', 'wb')
        data = []
        data.append(xnew)
        for i in range(num):
            data.append(f[i](xnew))
        np.save(file, data)
        file.close()

    if ret:
        return f


def load_data(plot=False, CSV=False, NPY=False, ret=False):
    """Loads plots and returns both .npy, .csv data formats that are stored for later use in testing GMM
     and GMR capabilities.
    Parameters
    ----------
    plot : bool (default: False)
        Decides if you want to plot the loaded data
    CSV : bool (default: False)
        Decides if you want to load the data stored as .csv format
    NPY : bool (default: False)
        Decides if you want to save the data sored as .npy format
    ret : bool (default: False)
        Decides if you want to return the loaded data as numpy parameter
    Returns
    -------
    data : array, shape (num, 100)
        If the parameter 'ret' is True that it return the generated data as npy array where num is the
        number of demonstrations.
    """
    data = []

    if NPY:
        data = np.load(str(pathlib.Path(__file__).parent.resolve()) + '/data.npy')
        # data = [container[key] for key in container]
        n, _ = np.shape(data)

    if plot:
        for i in range(n):
            if i != 0:
                plt.plot(data[i])
        plt.show()

    if ret:
        return data


if __name__ == "__main__":
    generate_data(num=10, plot=True, saveCSV=False, saveNPY=True, ret=False)
    data = load_data(plot=True, CSV=False, NPY=True, ret=True)
