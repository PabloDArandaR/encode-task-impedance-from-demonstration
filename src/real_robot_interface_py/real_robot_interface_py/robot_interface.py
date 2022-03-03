from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
import numpy as np
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

dataArray = Float32MultiArray

def torqueToSpeed(torque: np.array):
    pass

class robotInterface(Node):
    '''
    class robotInterface: Communicate with the real robot sending the commands and reading the different sensor values.
    '''
    def __init__(self):
        super().__init__("UR_interface")

        # Declare parameters
        self.declare_parameter("ip")
        self.declare_parameter("dt")

        # Declare 
        try:
            self.ip = self.get_parameter("ip").get_parameter_value().string_value
            print(f"IP is: {self.ip}")
        except TypeError as err:
            print(f"[ERROR] Not possible to set the parameter ip: {err}")
            sys.exit()
        try:
            self.dt = self.get_parameter("dt").get_parameter_value().double_value
            print(f"dt is: {self.dt}")
        except TypeError as err:
            print(f"[ERROR] Not possible to set the parameter dt: {err}")
            sys.exit()

        # Communication variables with UR
        self.control = RTDEControlInterface(self.ip)
        self.receive = RTDEReceiveInterface(self.ip)

        # Publishers
        self.action_subscriber = self.create_subscriber(dataArray, 'action', self.action_callback, 10)

        # Sensor reading functions
        self.sensor_timer = self.create_timer(dt, self.sensor_callback)
        self.torque_publisher = self.create_publisher(dataArray, 'force', 10)
        self.q_publisher = self.create_publisher(dataArray, 'q', 10)
        self.dq_publisher = self.create_publisher(dataArray, 'dq', 10)
    
    def action_callback(self, msg: dataArray):
        torque = np.reshape(np.array(msg.data), (6,1))
        speed = torqueToSpeed(torque)
        self.control.speedJ(speed)
        pass

    def sensor_callback(self):
        q = dataArray(data = self.receive.getActualQ())
        dq = dataArray(data = self.receive.getActualQd())
        torque = dataArray(data = self.receive.getJointTorques())

        self.q_publisher(q)
        self.dq_publisher(dq)
        self.torque_publisher(torque)
        pass

def main(args=None):
    rclpy.init(args=args)

    robot = robotInterface()
    rclpy.spin(robot)
    robot.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    
    main()