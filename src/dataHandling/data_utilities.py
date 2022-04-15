import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("src/")
import ur_kinematics.UR as kin

def combineDatasets(list_of_arrays: list):
    '''
    combine the datasets found in search_path. All must be the same shape. Stores the combined dataset.
    input 
        - list_of_arrays <list of np.array>: list of numpy arrays with the given iterations of the task
    output
        - combined <np.array>: all arrays combined in one
    '''
    list_of_arrays_copy = [np.concatenate([el, np.full((el.shape[0],1), i)], axis=1) for i, el in enumerate(list_of_arrays)]
    combined = np.concatenate(list_of_arrays_copy, axis=0)
    return combined

def eliminateTimeOffset(input: np.array):
    '''
    Eliminates the offset of time for the entire ordered dataset of samples from the same training iteration.
    input 
        - input <np.array mx19>: input array to eliminate the time offset
    output
        - output <np.array mx19>: array of combined input datasets
    '''
    output = np.copy(input)
    output[:,-1] -= output[0,-1]
    return output

def loadTaskSets(dir: str):
    '''
    Load all the datasets found in the directory used as input.
    input 
        - dir <str>: directory where datasets are stored.
    output
        - output <list of np.array>: list of arrays read from the directory
    '''
    output = [np.load(dir + el) for el in os.listdir(dir) if el[-4 :] == ".npy"]
    return output

def filterAmount(input: np.array, n: int):
    '''
    If a datasets doesn't have more than a given amount of rows, returns False.
    input 
        - input <np.array>: array to check whether to eliminate or not.
        - n <int>: minimum number of elements
    output
        - check <bool>: True if element is valid
    '''
    if input.shape[0] > n:
        return True
    else:
        return False

def plotTrajectory(input: np.array, t: str = ""):
    '''
    Plots the trajectory given by a dataset.
    input 
        - input <np.array>: input array with the trajectory.
        - t <str>: type of plot for titles. p for position, s for speed, f for force/torque.
    output
        - fig, ax <fix and x, matplotlib.pyplot> data of the plot
    '''
    if t == "p":
        dict_title = {0: "x position", 1: "y position", 2: "z position", 3: "axis-angle component x", 4: "axis-angle component y", 5: "axis-angle component z"}
    elif t == "s":
        dict_title = {0: "x velocity", 1: "y velocity", 2: "z velocity", 3: "axis-angle velocity x", 4: "axis-angle velocity y", 5: "axis-angle velocity z"}
    elif t == "f":
        dict_title = {0: "x force", 1: "y force", 2: "z force", 3: "x torque", 4: "y torque", 5: "z torque"}
    else:
        dict_title = {0: "", 1: "", 2: "", 3: "", 4: "", 5: ""}

    fig, axs = plt.subplots(3,2)
    for i in range(6):
        row = i % 3
        col = int(i / 3)
        axs[row, col].plot(input[:,i])
        axs[row, col].set_title(dict_title[i])

    return fig, axs

def combine(task_dir:str, n:int):
    '''
    Performs all the required actions on the datasets and returns the list of filtered datasets as well as the combined dataset that has all the iterations in the same set
    input 
        - task_dir <str>: the directory where all the iterations can be found.
        - n <int>: minimum number of elements that must exist in the iteration in order to take it into account
    output
        - list_datasets <list of np.array>: list of arrays of each of the training iterations.
        - dataset <np.array>: combined dataset
    '''
    list_datasets = loadTaskSets(task_dir)
    list_datasets = list(filter(lambda x: filterAmount(x, n), list_datasets))
    list_datasets = list(map(eliminateTimeOffset, list_datasets))
    dataset = combineDatasets(list_datasets)

    return list_datasets, dataset