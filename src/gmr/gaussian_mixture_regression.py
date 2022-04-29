import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.patches import Ellipse
from sklearn.mixture import BayesianGaussianMixture
from itertools import cycle
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
from gmr.utils import check_random_state
from generate_fake_data import load_data


def train_and_return_1D(X, gaus_num=10, out_dim=100):
    """This function is used as in easy to use interface for GMRpy and sklearn Bayesian Gaussian Mixture alorithm, but
    only for one dimension data plus time.
    Parameters
    ----------
    X : npy array, shape (n_demonstrations, n_steps, n_tasks_dim)
        Training data for the GMM and GMR algorithm. This function only allows one dimensional data.
    gaus_num : int (default: 10)
        The number of Gaussians that you;re going to try fit the data with
    out_dim : int (default: 100)
        The number of linearly separated values that work as normalized time
    Returns
    -------
    (time, out) : numpy arrays, shape ((out_dim,), (out_dim,))
        Returns two numpy arrays one with the predicted values and one with the normalized time values.
    """
    n_demonstrations, n_steps, n_task_dims = X.shape
    if n_task_dims != 1:
        raise ValueError("The training data was expected to be 1D and was received " + str(n_task_dims) + "D!")
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
    bgmm = BayesianGaussianMixture(n_components=n_components, max_iter=1000, random_state=random_state, tol=1e-7).fit(
        x_train)
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
    """This function is used as in easy to use interface for GMRpy and sklearn Bayesian Gaussian Mixture algorithm, and
        it works for multidimensional data plus time. The dimensions are treated separately from each other, so the
        final number of the Gaussians is going to be dimensions*gaus_num.
        Parameters
        ----------
        X : npy array, shape (n_demonstrations, n_steps, n_tasks_dim)
            Training data for the GMM and GMR algorithm. This function allows multiple dimensional data, and it's
            working based on the
        gaus_num : int (default: 10)
            The number of Gaussians that you;re going to try fit the data with
        out_dim : int (default: 100)
            The number of linearly separated values that work as normalized time
        Returns
        -------
        (time, out) : numpy arrays, shape ((out_dim,), (out_dim,n_tasks_dim))
            Returns two numpy arrays one with the predicted values and one with the normalized time values.
        """
    n_demonstrations, n_steps, n_task_dims = X.shape
    out = np.zeros((out_dim, n_task_dims))
    for i in range(n_task_dims):
        x_train = X[:, :, i]
        time, out[:, i] = train_and_return_1D(x_train[:, :, np.newaxis], gaus_num=gaus_num, out_dim=out_dim)
    return time, out


def train_and_return_PD_connected(X, gaus_num=10, out_dim=100):
    """This function is used as in easy to use interface for GMRpy and sklearn Bayesian Gaussian Mixture algorithm, and
            it works for multidimensional data plus time. The dimensions are NOT treated separately from each other, so
            the final number of the Gaussians is going to be only gaus_num, aand theid dimensionality is going to be
            n_tasks_dim + 1.
            Parameters
            ----------
            X : npy array, shape (n_demonstrations, n_steps, n_tasks_dim)
                Training data for the GMM and GMR algorithm. This function allows multiple dimensional data, and it's
                working based on the
            gaus_num : int (default: 10)
                The number of Gaussians that you;re going to try fit the data with
            out_dim : int (default: 100)
                The number of linearly separated values that work as normalized time
            Returns
            -------
            (time, out) : numpy arrays, shape ((out_dim,), (out_dim,n_tasks_dim))
                Returns two numpy arrays one with the predicted values and one with the normalized time values.
            """
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


if __name__ == '__main__':
    X = load_data(plot=False, NPY=True, ret=True)
    dem, len = X.shape
    X = X[:, :, np.newaxis]
    X = np.dstack((X, X))
    time, X_PD = train_and_return_PD(X=X, gaus_num=10, out_dim=100)
    print(X_PD.shape)
    plt.plot(X_PD)
    plt.show()
