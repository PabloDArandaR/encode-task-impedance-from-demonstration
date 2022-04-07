import sys
import os
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool, String

dataArray = Float32MultiArray

class controllerNode(Node):
    '''
    Controller node that handles the messages received from the robot and
    simulation and sends the required actions/speed based on the parameters
    obtained from the trained model.

    ...

    Attributes
    ----------
    q : np.array
        numpy array with the joint positions of the last step
    q_1 : np.array
        numpy array with the joint positions of the previous step
    dq : np.array
        numpy array with the joint velocities of the last step
    dq_1 : np.array
        numpy array with the joint velocities of the previous step
    tau : np.array
        numpy array with the torque of the last step
    checked : np.array
        list of booleans that indicate which of the variables have been checked
            -0: joint position
            -1: joint velocity
            -2: torque
    first : np.array
        list of booleans that indicate if is the first step for each of the variables
            -0: joint position
            -1: joint velocity
    action_model : TODO
        trained model that is used to estimate the parameters #TODO might be changed to a sequence type that stores the references poses, matrixes, ...

    Methods
    -------
    torque_callback(msg)
        Handles the reception of torque messages
    position_callback(msg)
        Handles the reception of position messages
    velocity_callback(msg)
        Handles the reception of velocity messages
    reset_callback(msg)
        Handles the reception of reset messages to reset the controller
    calculateAction()
        Calculates the action for the system (joint velocity due to restrictions with the system) based on the last received messages
    allChecked()
        Function that checks whether the messages for the last step have been all received
    '''
    def __init__(self):
        super().__init__("controller")

        # Class fields
        self.q = np.zeros((6,1))
        self.dq = np.zeros((6,1))
        self.q_1 = np.zeros((6,1))
        self.dq_1 = np.zeros((6,1))
        self.tau = np.zeros((6,1))
        self.checked = [False, False, False]
        self.first = [True, True]
        self.action_model = np.zeros((1,1)) # TODO: Here is where the model will be loaded
        self.calculate = True

        # Declare publishers and subscribers
        self.robot_subsriber = self.create_subscription(dataArray, '/ur/robot', self.robot_callback, 10)
        self.reset_subscriber = self.create_subscription(Bool, '/ur/reset', self.reset_callback, 10)
        self.mode_subscriber = self.create_subscription(String, '/mode', self.mode_callback, 10)
        self.training_subscriber = self.create_subscription(Bool, '/training', self.training_callback, 10)

        self.output_publisher = self.create_publisher(dataArray, '/ur/output_controller', 10)
    
    def robot_callback(self, msg):
        '''
        TODO: Callback function for the torque message.
        input:
            - msg <message data of type dataArray> : message with the torque information
        '''
        if self.calculate:
            self.tau = np.array(msg.data)
            self.checked[2] = True
            if self.allChecked():
                self.calculateAction()
    
    def mode_callback(self, msg):
        if msg.data == "G":
            self.output_publisher = None
            self.calculate = False
        if msg.data == "T":
            self.output_publisher = None
            self.calculate = False
        if msg.data == "C":
            self.output_publisher = self.create_publisher(dataArray, '/ur/output_controller', 10)
            self.calculate = True


    def reset_callback(self, msg):
        '''
        Callback function for the reset message that would reset the controller parameters.
        input:
            - msg <message data of type dataArray> : message with the torque information
        '''
        self.q = np.zeros((6,1))
        self.dq = np.zeros((6,1))
        self.q_1 = np.zeros((6,1))
        self.dq_1 = np.zeros((6,1))
        self.tau = np.zeros((6,1))
    
    def calculateAction(self): # TODO: Based on the real controller 
        '''
        Calculate the action based on the parameters that have been received in the last messages.
        '''
        M, D, K = self.action_model
        A = self.q + self.q_1 + self.dq + self.dq_1 + self.tau

        return A


def main(args=None):
    '''
    Main function that initializes the node.
    '''
    rclpy.init(args=args)

    controller = controllerNode()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    
    main()