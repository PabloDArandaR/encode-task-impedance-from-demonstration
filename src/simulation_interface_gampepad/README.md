## System Requirements
These files have been tested with:
- Ubuntu 20.04
- Nvidia GTX 3050ti
- ROS noetic version 1.15.14
- Python 3.8.10


## Excecution
 
To execute the ROS package, first make sure to include it in your ROS environment and install the libraries specified in requirements.txt. 

Inside the folder “scripts” there are two scripts, gamepad_publisher.py and store_data.py. 

The first one will publish the action using the gamepad. The left stick will control the X and Y axis, while the right stick will control the Z axis. To move faster keep the left bumper pressed while, to move slower keep the right bumper pressed. For changing to orientational manipulation, keep the left thumb pressed. 

For moving to another iteration, press “A”. To finish the simulation and save the data, press “Y”. The file will be saved in the folder specified inside of “store_data.py”.
In order to save the data, make sure to have the simulation in coppeliasim running.

In order to execute the ros package:
```bash
roscore
```

```bash
rosrun simulation_interface_gampepad gamepad_publisher.py
```

```bash
rosrun simulation_interface_gampepad store_data.py
```
