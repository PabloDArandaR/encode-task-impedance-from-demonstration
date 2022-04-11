import numpy as np
import sys
import os

sys.path.append("src/")
import ur_kinematics.UR as kin

def combineData(list_of_arrays: list):
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
    combine the datasets found in search_path. All must be the same shape. Stores the combined dataset.
    input 
        - input <np.array mx16>: input array to eliminate the time offset
    output
        - output <np.array mx16>: array of combined input datasets
    '''
    output = np.copy(input)
    output[:,-1] -= output[0,-1]
    return output

def loadTaskSets(dir: str):
    '''
    Take the entire dataset and transform the joint position, speed, and torque with DK.
    input 
        - dir <str>: directory where datasets are stored.
    output
        - output <list of np.array>: list of arrays read from the directory
    '''
    output = [np.load(dir + el) for el in os.listdir(dir) if el[-4 :] == ".npy"]
    return output

def filterAmount(input: np.array, n: int):
    '''
    Take the entire dataset and transform the joint position, speed, and torque with DK.
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