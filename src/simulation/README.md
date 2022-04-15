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
./coppeliaSim.sh $DOWNLOAD_PATH/encode-task-impedance-from-demonstration/src/simulation/scene/ur5_simulation.ttt
```

## ROS Topics
The simulation publish in four topics:
- The topic "/ur5_simulation/jointConfig" contains the position of joints in the simulation.
- The topic "//ur5_simulation/ToolPosition" contains the position of the tool in the simulation.
- The topic "/ur5_simulation/torques" contains the torques of the robot's joints in the simulation.
- The topic "/ur5_simulation/speeds" contains the velocity of the robot's joints in the simulation.

In order to change the position, the simulation is listening to "/move_command"
In order to do another iteration, the simulation is listening to "/ur5_simulation/reset"
