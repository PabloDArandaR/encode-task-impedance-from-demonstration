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
    def __init__(self, ip: str, dt: float):
        super().__init__("UR_interface")
        self.ip = ip
        print(f"IP is: {self.ip}")
        self.dt = dt
        print(f"dt is: {self.dt}")

        self.control = RTDEControlInterface(self.ip)
        self.receive = RTDEReceiveInterface(self.ip)
        #self.io      = RTDEIOInterface(self.ip)

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

def main(ip, dt = 0.01, args=None):
    rclpy.init(args=args)

    robot = robotInterface(ip, dt)
    rclpy.spin(robot)
    robot.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    if len(sys.argv) < 3:
        print("[ERROR] Not enough arguments")
        sys.exit()
    
    ip = sys.argv[1]
    dt = float(sys.argv[2])

    main(ip, dt)