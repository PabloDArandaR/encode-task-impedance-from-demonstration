import sys
import os
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool

dataArray = Float32MultiArray

class controllerNode(Node):
    ''' Class definition of the controller node'''
    def __init__(self):
        super().__init__("controller")

        self.q = np.zeros((6,1))
        self.dq = np.zeros((6,1))
        self.q = np.zeros((6,1))
        self.dq = np.zeros((6,1))
        self.tau = np.zeros((6,1))
        self.first = True
        self.action_model = np.zeros((1,1)) # TODO: Here is where the model will be loaded
        self.time_wrapper = np.zeros((1,1)) # TODO: Time wrapper be able to determine how much of the path has been done

        # Declare publishers and subscribers
        self.input_subscriber = self.create_subscription(dataArray, 'robot_input', self.input_callback, 10)
        self.reset_subscriber = self.create_subscription(Bool, 'first_input', self.reset_callback, 10)

        self.action_publisher = self.create_publisher(dataArray, 'action', 10)

    def input_callback(self, msg):
        if self.first:
            self.first = False
            self.q_1 = np.array(msg.data[:6])
            self.dq_1 = np.array(msg.data[6:12])
        else:
            self.q_1 = np.copy(self.q)
            self.dq_1 = np.copy(self.dq)
        
        self.q   = np.array(msg.data[:6])
        self.dq  = np.array(msg.data[6:12])
        self.tau = np.array(msg.data[6:12])

        msg = dataArray(data=self.calculateAction())
        self.action_publisher(msg)
    
    def reset_callback(self, msg):
        self.first = True
    
    def calculateAction(self): # TODO: Based on the real controller strucurs, time wrapper class, and the estimation model for the different matrixes
        sigma = self.time_wrapper
        M, D, K = self.action_model
        A = self.q + self.q_1 + self.dq + self.dq_1 + self.tau

        return A


def main(args=None):
    rclpy.init(args=args)

    controller = controllerNode()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    
    main()