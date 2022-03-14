## System Requirements
For running the simulation you'll need:
- coppeliaSim V4.1.0
- Vortex physics engine
- ROS melodic

## Excecution
In order to excecute the simulation, first, you'll need to start ROS core
```bash
roscore
```
And then start the simulation openning the scene
```bash
./coppeliaSim.sh $DOWNLOAD_PATH/encode-task-impedance-from-demonstration/simulation/scene/ur3_simulation.ttt
```

## ROS Topics
The simulation publish in three topics:
- The topic "/ur3/joint_positions" contains the position of the robot in the simulation.
- The topic "/ur3/joint_torques" contains the torques of the robot's joints in the simulation.
- The topic "/ur3/joint_velocities" contains the velocity of the robot's joints in the simulation.

In order to change the position, the simulation is listening to "/ur3/move_command"
