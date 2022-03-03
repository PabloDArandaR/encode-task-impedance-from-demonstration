# Encode Task Impedance from Demonstration
Project in Advanced Robotics course project at SDU 21/22. Implementation of  learning method for skills for arm robots based on GMM with Rieamannian Manifolds

## Initializing each node

To execute the <code>real_robot_interface_py</code> command the following command should be used.

    ros2 run real_robot_interface_py interface --ros-args -p dt:=<sampling_time> -p ip:=<robot/ip>
