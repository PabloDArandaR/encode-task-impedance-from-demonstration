import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.patches import Ellipse
from sklearn.mixture import BayesianGaussianMixture
from itertools import cycle
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
from gmr.utils import check_random_state


def train_and_return_1D(X, gaus_num=10, out_dim=100):
    n_demonstrations, n_steps, n_task_dims = X.shape
    if n_task_dims != 1:
        print("The training data was expected to be 1D and was received "+str(n_task_dims)+"D!")
        return None
    x_train = np.empty((n_demonstrations, n_steps, n_task_dims + 1))
    x_train[:, :, 1:] = X
    t = np.linspace(0, 1, n_steps)
    x_train[:, :, 0] = t
    x_train = x_train.reshape(n_demonstrations * n_steps, n_task_dims + 1)
    random_state = check_random_state(0)
    n_components = gaus_num
    initial_means = kmeansplusplus_initialization(x_train, n_components, random_state)
    initial_covs = covariance_initialization(x_train, n_components)
    bgmm = BayesianGaussianMixture(n_components=n_components, max_iter=300, random_state=random_state).fit(x_train)
    gmm = GMM(
        n_components=n_components,
        priors=bgmm.weights_,
        means=bgmm.means_,
        covariances=bgmm.covariances_,
        random_state=random_state)
    t_test = np.linspace(0, 1, out_dim)
    output = gmm.predict(np.array([0]), t_test[:, np.newaxis])
    return t_test, output.flatten()

def train_and_return_PD(X, gaus_num=10, out_dim=100):
    n_demonstrations, n_steps, n_task_dims = X.shape
    out = np.zeros((out_dim, n_task_dims))
    for i in range(n_task_dims):
        x_train = X[:, :, i]
        time, out[:, i] = train_and_return_1D(x_train[:, :, np.newaxis], gaus_num=gaus_num, out_dim=out_dim)
    return time, out

def train_and_return_PD_connected(X, gaus_num=10, out_dim=100):
    n_demonstrations, n_steps, n_task_dims = X.shape
    x_train = np.empty((n_demonstrations, n_steps, n_task_dims + 1))
    x_train[:, :, 1:] = X
    t = np.linspace(0, 1, n_steps)
    x_train[:, :, 0] = t
    x_train = x_train.reshape(n_demonstrations * n_steps, n_task_dims + 1)
    random_state = check_random_state(0)
    n_components = gaus_num
    initial_means = kmeansplusplus_initialization(x_train, n_components, random_state)
    initial_covs = covariance_initialization(x_train, n_components)
    bgmm = BayesianGaussianMixture(n_components=n_components, max_iter=300, random_state=random_state).fit(x_train)
    gmm = GMM(
        n_components=n_components,
        priors=bgmm.weights_,
        means=bgmm.means_,
        covariances=bgmm.covariances_,
        random_state=random_state)
    t_test = np.linspace(0, 1, out_dim)
    output = gmm.predict(np.array([0]), t_test[:, np.newaxis])
    return t_test, output.squeeze()

if __name__=='__main__':
    from generate_fake_data import load_data
    X = load_data(plot=True, NPY=True, ret=True)
    dem, len = X.shape
    print(X.shape)
    X = X[:, :, np.newaxis]
    X = np.dstack((X, X))
    print(X.shape)
    time, X_1D = train_and_return_PD_connected(X=X, gaus_num=10, out_dim=100)
    print(X_1D.shape)
    plt.plot(X_1D)
    plt.show()