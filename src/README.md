# Source code folder

The different elements that compound the project are stored in each of the folders correspondent to it. About the scripts found in the source folder:

## Gui training script

Handles the communication and control of the robot during the training and obtention of learning iterations for a given skill. It has 2 input arguments:
  - dt: sampling time in seconds expressed as a floating point number.
  - task: task ID.

## Offline pipeline

Script that uses the given functions in the modules to obtain the model that is later used for the controller.