# Encode Task Impedance from Demonstration
Project in Advanced Robotics course project at SDU 21/22. Implementation of  learning method for skills for arm robots based on GMM with Rieamannian Manifolds

## Communication topics

To perform the communication between the different ROS nodes for both the simulation environment and the real robot, various topics are defined:

Those involved in the information received from the robot system are:

 - Robot joint torque: <code> /ur/joint_torque </code>
 - Robot joint position: <code> /ur/joint_torque </code>
 - Robot joint velocity: <code> /ur/joint_velocity </code>

Those involved in the information to be sent to the robot system by the controller are:
 - Robot joint position: <code> /controller/joint_position </code>
 - Robot joint velocity: <code> /controller/joint_velocity </code>

## Initializing each node

To execute the <code>real_robot_interface_py</code> command the following command should be used.

    ros2 run real_robot_interface_py interface --ros-args -p dt:=<sampling_time> -p ip:=<robot/ip>
