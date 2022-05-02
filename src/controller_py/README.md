# Controller design
<p align="center">
    <img src="Pictures/v1_1.png" width="600">
</p>
<center>Controller structure</center>

The controller will perfom the following actions:
- Detect the part of the path in which the robot is, according to the time wrapping module.
- Estimate the reference pose for that step
- Estimate the M, D, K matrixes of the controller based on the model found through GMM, GMR, ...
- Use all the previous parameters and the current output to calculate the required action for the controller.

## ROS subscribers and publishers (**TODO**)
