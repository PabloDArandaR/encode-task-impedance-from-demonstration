import sys
import os
import numpy as np

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
        print(f"Shape of new_data: {new_data.shape}")
        print(f"Shape of offset_array: {offset_array.shape}")
        print(offset_array)

        new_data[:,offset_index] += offset_array[:,0]
        data = np.concatenate((data, new_data), axis=0)
        offset = np.max(data[:,offset_index]) + 1

    np.save(store_path + filename, data)
    return data

if __name__=="__main__":
    shape = (50,10)
    new_data_path = os.getcwd() + "/src/dataHandling/"
    store_path = os.getcwd() + "/src/dataHandling/"
    print(f"Current wd: {os.getcwd()}")

    for i in range(5):
        new_set = np.random.rand(shape[0], shape[1])
        new_set[:,shape[1] - 1] = 0
        np.save(new_data_path + "test" + str(i) + ".npy", new_set)

    combineData(search_path=new_data_path, store_path=store_path, filename="def.npy", offset_index=shape[1] - 1, n_cols=shape[1])

