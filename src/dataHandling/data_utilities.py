import numpy as np
import sys
import os

sys.path.append("src/")
import ur_kinematics.UR as kin

def deleteLastIter(data: np.array):
    '''
    Delete the last iteration given the input file. The last iteration will allways be the last column of the array.
    input 
        - data <numpy array>: numpy array with the data.
    output
        - output <numpy array>: numpy array with the filtered.
    '''
    max_value = np.max(data[:,-1])
    output = np.delete(data, np.where(data[:,-1] == max_value), axis=0)
    return output

def combineData(search_path: str, store_path: str, filename:str, offset_index: int, n_cols:int):
    '''
    combine the datasets found in search_path. All must be the same shape. Stores the combined dataset.
    input 
        - search_path <str>: path where the datasets should be found
        - store_path <str>: path where the combined dataset will be written
        - filename <str>: name of the combined dataset file
        - offset_index<int>: index in which the number of the iteration is stored
        - n_cols <int>: number of columns expected in the dataset
    output
        - data <sum of rows of input files x n_cols, numpy array>: array of combined input datasets
    '''
    offset = 0
    data = np.zeros((0, n_cols))
    dirs = os.listdir(search_path)
    for file in [dir for dir in os.listdir(search_path) if dir[-4:] == ".npy"]:
        new_data = np.load(search_path + file)
        if new_data.shape[1] != n_cols:
            print(f"[ERROR] Expected number of columns ({n_cols}) different to the number of columns in dataset {file}")
            sys.exit()
        offset_array = np.full([new_data.shape[0], 1], fill_value= offset)

        new_data[:,offset_index] += offset_array[:,0]
        data = np.concatenate((data, new_data), axis=0)
        offset = np.max(data[:,offset_index]) + 1

    np.save(store_path + filename, data)
    return data

def transformDK(path_to_file:str, store = False, path_to_savefile = "resources/transformedDK.npy"):
    '''
    Take the entire dataset and transform the joint position, speed, and torque with DK.
    input 
        - path_to_file <str>: path to dataset file.
        - store <bool>: default False. Store the dataset.
        - path_to_savefile <str>: path to savefile.
    output
        - data <sum of rows of input files x n_cols, numpy array>: transformed array
    '''
    input = np.load(path_to_file)
    Ta, Td, Talpha = kin.precomputeT()
    x = np.zeros((input.shape[0],7))
    dx = np.zeros((input.shape[0],6))
    for i in range(input.shape[0]):
        theta = input[i,:6]
        dtheta = input[i,6:12]
        _,T = kin.TfromConfig(theta, Ta, Td, Talpha)
        Jo, Jp = kin.Jacobian(T)

        x[i,:] = kin.transQuat(theta, Ta, Td, Talpha, normalize=True)
        v,w = kin.velocity(Jp=Jp[5,:,:], Jo=Jo[5,:,:], dq = dtheta)
        dx[i,:] = np.concatenate([v,w])
    
    data = np.concatenate([x,dx, np.reshape(input[:,12], (input.shape[0],1))], axis=1)

    if store:
        np.save(path_to_savefile, data)
    
    return data