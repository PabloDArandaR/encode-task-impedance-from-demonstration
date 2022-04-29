# Data handling

Functions for handling the data obtained from the task learning iterations. Some considerations about how the data is handled and how it is received from the UR system before getting in depth:
  - The UR robot returns returns the orientation in a compact angle-axis representation. This is, an array of 3 values, in which the modulus of the normalized vector gives the direction of rotation, and the modulus is the angle of rotation in radians.
  - The time column of the datasets is the amount of seconds passed since epoch, being a common reference for all of the points.

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
  - ```aarToAngleVector```: Transforms a given compact representation of axis-angle representation to a (vector, angle) representation.
    - inputs: 
        - numpy array of 1x3
    - outputs:
        - numpy array of 1x3
        - float
  - ```aarToQuat```: Transforms a given compact representation of axis-angle representation to quaternion representation
    - inputs: 
        - numpy array of 1x3
    - outputs:
        - numpy array of 1x4
  - ```transformToQuat```: Takes a demonstration set and modifies it to put the quaternions where the axis-angle representation appears
    - inputs: 
        - numpy array of nx19
    - outputs:
        - numpy array of nx20
  - ```combine```: Performs all the required actions on the datasets and returns the list of filtered datasets as well as the combined dataset that has all the iterations in the same set.
    - inputs:
        - str
        - int
    - outputs:
        - list of numpy arrays of nx20 (position (7 values), speed(6 values), forces (6 values), time)
        - numpy array of nx21 (position (7 values), speed(6 values), forces (6 values), time, ID demonstration)

## Import python modules


```python
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
```

## Functions involved in the loading of a dataset

### **Loading the dataset**

Firstly, the files with the data information must be loaded. By using the  ```loadTaskSets``` function, if the files are stored in the directory ``` resources/training_data/task_1/```, the function can be used the following way to load the data:

```python
output = loadTaskSets("resources/training_data/task_1/")
```

This will return a list of numpy arrays. Each arrays contains the samples for each of the demonstrations  of the task.

### **Filtering datasets by amount.**

The ```filterAmount``` function is used to check whether the amount of samples obtained for a given demonstration is enough or not. It receives 2 input parameters, the array that is going to be checked and the minimum amount of samples that the demonstration must contain. It will return True in case the the demonstration is valid and False elsewhere.

The usage of the function is described as it follows, by using the first dataset of the previously loaded datasets and a minimum amount of 10 samples:
```python
filterAmount(output[0], 10)
```

### **Transforming the dataset angle values from angle-axis description to quaternions**

The values obtained of orientation from the ur-rtde driver of position in cartesian space are in angle-axis representation. In order to change this values to quaternions, the function ```transformToQuat``` is used. The function receives an entire demonstration as input and returns an array with the same number of rows and an extra column. This takes into consideration the expected shape of the input array (6 columns for position, 6 columns for speed, 6 columns for torque/momentum, and 1 column for time) and returns an array with the following structure: 7 columns for position, 6 columns for speed, 6 columns for torque/momentum, and 1 column for time.

Inside, it uses the function ```aarToQuat```. The function receives a numpy array of 3 columns and 1 row and returns a numpy array of 4 columns and 1 row (```[q_w, q_x, q_y, q_z]```). The functions uses inside the function ```aarToAngleVector```, which transforms the compact representation of axis-angle representation (a vector of 3 values), to a (vector, angle) representation, in which the axis of rotation and the angle rotation are explicitily returned.  Example of usage of the aarToQuat function:

```python
>>> input = np.array([2.0, 1.0, 2.0])
>>> aarToQuat(input)
array([[0.0707372 , 0.66499666, 0.33249833, 0.66499666]])
```
### **Eliminate the time offset**

The function ```eliminateTimeOffset``` will eliminate the offset existent in the time variable by updating the value in the time column to the difference between the original value minus the value found in the first position in the input demonstration. The function, used again in the first element of the loaded datasets, is demonstrated as:

```python
output[0] = eliminateTimeOffset(output[0])
```

### **Combine all the datasets**

To combine all the datasets, the function ```combineDatasets``` is used. The function will add an additional column that identifies each of the datasets with a number. After that, it will create a new dataset that consists of all the demonstrations with their identification number concatenated in the rows direction.

## Full demonstration

This section will demonstrate each of the steps of the function ```combine```, which uses the previously demonstrated functions to return the processed demonstrations of a given task in 2 different formats:
  - A list of numpy arrays, where each array is a processed demonstration of the task.
  - A numpy array, in which all the demonstrations are combined into 1 dataset.

The function receives 2 inputs: the directory where the task demonstrations are stored, and the minimum amount of samples that a demonstration must contain. An example of usage of the combine function:
```python
>>> task_dir = "resources/training_data/task_1/"
>>> min_demos = 10
>>> list_datasets, dataset = combine(task_dir, min_demos)
```

## Step by step:

The names that are given to the inputs are: ```task_dir``` and ```n```.

Firstly, the loading of the dataset is done through the ```loadTaskSets``` function.

```python
list_datasets = loadTaskSets(task_dir)
```

The list of demonstrations is then filtered by using the output of the ```filterAmount``` function applied to each of the elements of the list by using the ```filter``` function of Python. A lambda expression is used to fix the minimum number of elements that the ```filterAmount``` function requires to the input ```n```. Since the output of ```filter``` is an iterable, the output is then transformed into a list.

```python
list_datasets = list(filter(lambda x: filterAmount(x, n), list_datasets))
```

Once the demonstrations are filtered by the amount of samples, the rotational part of each of the demonstrations is transformed into quaternion format by using the ```transformToQuat```. The ```map``` functions is used to apply it each of the demonstrations and, as in the previous case, the output will be transformed into a list, since ```map``` returns an iterable.

```python
list_datasets = list(map(transformToQuat, list_datasets))
```

The offset in time is then eliminated with the ```elminateTimeOffset``` function. Again, the ```map``` function is used to apply it to every element of the list of demonstrations and its output transformed to a list.

```python
list_datasets = list(map(eliminateTimeOffset, list_datasets))
```

The demonstrations are then combined and returned, both the list of demonstrations and the combined demonstrations dataset:
```python
dataset = combineDatasets(list_datasets)

return list_datasets, dataset
```