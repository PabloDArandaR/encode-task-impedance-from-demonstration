# Data handling

Functions for handling the data obtained from the task learning iterations.

## Functions

  - ```combineDatasets```: combine the datasets found in search_path. All must be the same shape. Stores the combined dataset.
    - inputs:
        -list of numpy arrays.
    - output:
        - numpy array.
  - ```eliminateTimeOffset```: Eliminates the offset of time for the entire ordered dataset of samples from the same training iteration.
    - inputs:
        - numpy array of nx19
    - outputs:
        - numpy array of nx19
  - ```loadTaskSets```: Load all the datasets found in the directory used as input.
    - inputs:
        - str
    - outputs:
        - list of numpy arrays
  - ```filterAmount```: If a datasets doesn't have more than a given amount of rows, returns False.
    - inputs:
        - numpy array of nx19
        - int
    - outputs:
        - bool
  - ```plotTrajectory```: Plots the trajectory given by a dataset.
    - inputs: 
        - numpy array of nx19
        - str
    - outputs:
        - fig and ax of matplotlib.pyplot
  - ```combine```: Performs all the required actions on the datasets and returns the list of filtered datasets as well as the combined dataset that has all the iterations in the same set.
    - inputs:
        - str
        - int
    - outputs:
        - list of numpy arrays
        - numpy array of nx19

