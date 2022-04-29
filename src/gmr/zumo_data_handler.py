import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from gaussian_mixture_regression import *


def load_zumo_data():
    position = np.load(
        str(pathlib.Path(__file__).parent.resolve()) + '/../../resources/parameters/task_up/xdesired.npy')
    est_param = np.load(
        str(pathlib.Path(__file__).parent.resolve()) + '/../../resources/parameters/task_up/estparam.npy')
    x_pos = position[1, :]
    y_pos = position[2, :]
    z_pos = position[3, :]
    total_time = position[0, 99]
    pos = np.vstack((x_pos, y_pos, z_pos))
    pos2 = np.zeros([3, 5, 100])
    for i in range(5):
        pos2[:, i, :] = pos
    # print(pos2.shape)
    # print(est_param.shape)
    final_data = np.ones([3, 5, 100, 7])
    final_data[:, :, :, 0] = pos2
    final_data[:, :, :, 1:7] = est_param
    return total_time, final_data


if __name__ == '__main__':
    time_length, total_data = load_zumo_data()
    print(total_data.shape)
    dimen, dem, frec, data = total_data.shape
    for i in range(dimen):
        time, predict = train_and_return_PD(X=total_data[i, :, :, :], gaus_num=15, out_dim=100)
        for j in range(data):
                for k in range(dem):
                    plt.plot(total_data[i, k, :, j], 'k')
                plt.plot(predict[:, j], 'r')
                plt.show()
