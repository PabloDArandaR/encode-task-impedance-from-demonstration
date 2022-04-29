from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
import numpy as np
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, Empty
#from custom_msg_srv.srv import SensorCall

dataArray = Float64MultiArray

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
        if self.dt == None:
            print(f"[ERROR] Incorrect dt input: ({self.dt})")
            sys.exit()

        # Subscribers
        self.controller_subscriber = self.create_subscription(dataArray, '/ur/output_controller', self.position_callback, 10)
        self.gui_subscriber = self.create_subscription(dataArray, '/gui/position', self.gui_callback, 10)
        self.teach_subscriber = self.create_subscription(Bool, '/gui/teach', self.teach_callback, 10)
        self.zero_sensor_subcriber = self.create_subscription(Empty, '/reset_sensor', self.reset_callback,10)

        # Publishers
        self.request_publisher = self.create_publisher(dataArray, "/sensor_data",10)
    
        # Timer
        self.timer = self.create_timer(self.dt, self.sensor_callback)

        # Communication variables with UR
        self.control = RTDEControlInterface(self.ip)
        self.receive = RTDEReceiveInterface(self.ip)

    def position_callback(self, msg):
        self.control.moveL(msg.data[0:6])

    def gui_callback(self, msg):
        self.control.moveJ(msg.data[0:6])
    
    def teach_callback(self,msg):
        if msg.data == True:
            self.control.teachMode()
        elif msg.data == False:
            self.control.endTeachMode()
    
    def reset_callback(self, msg):
        self.control.zeroFtSensor()
        
    def sensor_callback(self):
        msg = dataArray()
        epoch = time.time()
        msg.data = self.receive.getActualTCPPose() + self.receive.getActualTCPSpeed() + self.receive.getActualTCPForce() + self.receive.getActualQ() + self.receive.getActualQd() + self.receive.getFtRawWrench() + self.control.getJointTorques() + [epoch]
        self.request_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    robot = robotInterface()
    rclpy.spin(robot)
    robot.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()