import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from gaussian_mixture_regression import *


def load_real_data(filepath: str, plot: bool):
    est_param = np.load(filepath)

    time = est_param[:, 0, :]
    force_encoder = np.linalg.norm(est_param[:, 22:25, :]) + 1e-16
    force = est_param[:, 22:25, :]
    timed_data = est_param[:, 1:22, :]

    force_unique = []
    for i in range(3):
        force_unique.append(np.unique(force[:, i, :]))
    force_unique = np.array(force_unique, dtype=object)

    # print(time.shape)
    # print(force.shape)
    # print(timed_data.shape)
    # print(np.shape(force_unique[0]))
    # print(np.shape(force_unique[1]))
    # print(np.shape(force_unique[2]))

    dem, dim, sample = timed_data.shape

    if plot:
        for i in range(1, 25):
            # print(force[:, i, :])
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            plt.title(str(i))
            ax1.scatter(time.transpose(), force[:, 0, :].transpose(), est_param[:, i, :].transpose())
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            ax2.scatter(time.transpose(), force[:, 1, :].transpose(), est_param[:, i, :].transpose())
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            ax3.scatter(time.transpose(), force[:, 2, :].transpose(), est_param[:, i, :].transpose())
            ax3.set_xlabel('time')
            ax3.set_ylabel('force')
            ax3.set_zlabel('dimension')
            ax2.set_xlabel('time')
            ax2.set_ylabel('force')
            ax2.set_zlabel('dimension')
            ax1.set_xlabel('time')
            ax1.set_ylabel('force')
            ax1.set_zlabel('dimension')
            plt.show()

    dict_x = {}
    for i in force_unique[0]:
        dict_x[i.item()] = []
    for sam in range(sample):
        for d in range(dem):
            dict_x[force[d, 0, sam].item()].append(timed_data[d, :, sam])

    dict_y = {}
    for i in force_unique[1]:
        dict_y[i.item()] = []
    for sam in range(sample):
        for d in range(dem):
            dict_y[force[d, 1, sam].item()].append(timed_data[d, :, sam])

    dict_z = {}
    for i in force_unique[2]:
        dict_z[i.item()] = []
    for sam in range(sample):
        for d in range(dem):
            dict_z[force[d, 2, sam].item()].append(timed_data[d, :, sam])

    forced_data_x = np.zeros([2, dim, len(force_unique[0])])
    forced_data_y = np.zeros([2, dim, len(force_unique[1])])
    forced_data_z = np.zeros([2, dim, len(force_unique[2])])

    for i in force_unique[0]:
        ndict = np.array(dict_x[i])
        forced_data_x[0, :, np.where(force_unique[0] == i)] = np.mean(ndict, axis=0)
        forced_data_x[1, :, np.where(force_unique[0] == i)] = np.mean(ndict, axis=0)

    for i in force_unique[1]:
        ndict = np.array(dict_y[i])
        forced_data_y[0, :, np.where(force_unique[1] == i)] = np.mean(ndict, axis=0)
        forced_data_y[1, :, np.where(force_unique[1] == i)] = np.mean(ndict, axis=0)

    for i in force_unique[2]:
        ndict = np.array(dict_z[i])
        forced_data_z[0, :, np.where(force_unique[2] == i)] = np.mean(ndict, axis=0)
        forced_data_z[1, :, np.where(force_unique[2] == i)] = np.mean(ndict, axis=0)

    # print(forced_data_x.shape)
    # print(forced_data_y.shape)
    # print(forced_data_z.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))

    for i in range(3):
        force_unique[i] = scaler.fit_transform(force_unique[i][:, np.newaxis])

    return time, force, est_param[:, 1:22, :].swapaxes(1, 2), forced_data_x.swapaxes(1, 2), \
           forced_data_y.swapaxes(1, 2), forced_data_z.swapaxes(1, 2), force_unique, force_encoder


if __name__ == '__main__':
    data_path = str(pathlib.Path(__file__).parent.resolve()) + '/../../resources/parameters/task_up_combinedweight/estimatedparams2.npy'
    time, force, the_data, fx, fy, fz, f_list, fe = load_real_data(filepath=data_path, plot=False)
    print(time.shape)
    print(force.shape)
    print(the_data.shape)
    print(fx.shape)
    print(fy.shape)
    print(fz.shape)
    print(f_list[2][:, np.newaxis].shape)

    demon, samples, dim = the_data.shape

    print('TRAINING STARTED')

    # encoder_d, the_data_predict = train_and_return_PD_connected(X=the_data, gaus_num=15, out_dim=100)
    # encoder_fx, fx_predict = train_and_return_PD_connected(X=fx, gaus_num=15, out_dim=100)
    # encoder_fy, fy_predict = train_and_return_PD_connected(X=fy, gaus_num=15, out_dim=100)
    # encoder_fz, fz_predict = train_and_return_PD_connected(X=fz, gaus_num=15, out_dim=100)

    timed_GMM = train_GMM(X=the_data, gaus_num=15, filename='timed_GMM')
    fx_GMM = train_GMM(X=fx, gaus_num=15, filename='fx_GMM')
    fy_GMM = train_GMM(X=fy, gaus_num=15, filename='fy_GMM')
    fz_GMM = train_GMM(X=fz, gaus_num=15, filename='fz_GMM')

    timed_GMM = load_GMM(filepath=str(pathlib.Path(__file__).parent.resolve()), filename='timed_GMM')

    time = np.linspace(0, 10, 300)

    for i in time:
        ni = i/10
        out = predict_GMR(gmm=timed_GMM, timestamp=ni)
        plt.scatter(i, out[5])
        #print(out.shape)
    plt.show()

    print(encoder_d.shape)
    print(encoder_fx.shape)
    print(encoder_fy.shape)
    print(encoder_fz.shape)

    print('SHAPES OF THE PREDICTION')
    print(the_data_predict.shape)
    print(fx_predict.shape)
    print(fy_predict.shape)
    print(fz_predict.shape)

    _, fx_samples, _ = fx.shape
    _, fy_samples, _ = fy.shape
    _, fz_samples, _ = fz.shape

    for i in range(dim):
        plt.subplot(411)
        for j in range(demon):
            plt.plot(encoder_d, the_data[j, :, i].transpose(), 'k')
        plt.plot(encoder_d, the_data_predict[:, i], 'r')

        plt.subplot(412)
        for j in range(fx_samples):
            plt.scatter(f_list[0][j], fx[0, j, i], c='k')
        plt.plot(encoder_fx, fx_predict[:, i], 'r')

        plt.subplot(413)
        for j in range(fy_samples):
            plt.scatter(f_list[1][j], fy[0, j, i], c='k')
        plt.plot(encoder_fy, fy_predict[:, i], 'r')

        plt.subplot(414)
        for j in range(fz_samples):
            plt.scatter(f_list[2][j], fz[0, j, i], c='k')
        plt.plot(encoder_fz, fz_predict[:, i], 'r')
        plt.show()
