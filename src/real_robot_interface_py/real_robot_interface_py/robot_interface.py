from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
import numpy as np
import sys
import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool, String
from custom_msgs_srvs.srv import SensorCall

dataArray = Float32MultiArray

class robotInterface(Node):
    ''' class robotInterface: Communicate with the real robot sending the commands and reading the different sensor values.'''
    def __init__(self):
        super().__init__("robot_interface")

        # Declare parameters
        self.declare_parameter("ip")
        self.declare_parameter("dt")

        # Read parameters from command line
        self.ip = self.get_parameter("ip").get_parameter_value().string_value
        if self.ip == '':
            print(f"[ERROR] Incorrect IP input: ({self.ip})")
            sys.exit()
        
        self.dt = self.get_parameter("dt").get_parameter_value().double_value
        if self.dt == 0:
            print(f"[ERROR] Incorrect sampling time input: ({self.dt})")
            sys.exit()


        # Subscribers
        self.controller_subscriber = self.create_subscription(dataArray, '/ur/output_controller', self.position_callback, 10)
        self.gui_subscriber = self.create_subscription(dataArray, '/gui/position', self.gui_callback, 10)

        # Services
        self.sensor_request = self.create_service(SensorCall, "/ur/sensor", self.sensor_callback)

        # Communication variables with UR
        self.control = RTDEControlInterface(self.ip)
        self.receive = RTDEReceiveInterface(self.ip)

    def sensor_callback(self, request, response):
        data_list = self.receive.getActualQ() + self.receive.getActualQd() + self.control.getJointTorques()
        response.data = data_list
        return response

    def position_callback(self, msg):
        self.control.moveL(msg.data[0:6])

    def gui_callback(self, msg):
        self.control.moveJ(msg.data[0:6])

def main(args=None):
    rclpy.init(args=args)
    robot = robotInterface()
    rclpy.spin(robot)
    robot.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()