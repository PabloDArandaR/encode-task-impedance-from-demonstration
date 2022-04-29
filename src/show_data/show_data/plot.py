from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
import numpy as np
import sys
import time
import random
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, Float64

dataArray = Float64MultiArray

max_samples = 500

class plt_class(Node):

    def __init__(self):
        super().__init__('plot_data')

        self.subs = self.create_subscription(dataArray, "/sensor_data", self.sensor_callback, 10)

        # Publishers
        self.publish_fx = self.create_publisher(Float64, "/fx",10)
        self.publish_fy = self.create_publisher(Float64, "/fy",10)
        self.publish_fz = self.create_publisher(Float64, "/fz",10)
        self.publish_mx = self.create_publisher(Float64, "/mx",10)
        self.publish_my = self.create_publisher(Float64, "/my",10)
        self.publish_mz = self.create_publisher(Float64, "/mz",10)
        self.publish_x = self.create_publisher(Float64, "/x",10)
        self.publish_y = self.create_publisher(Float64, "/y",10)
        self.publish_z = self.create_publisher(Float64, "/z",10)
        self.publish_rx = self.create_publisher(Float64, "/rx",10)
        self.publish_ry = self.create_publisher(Float64, "/ry",10)
        self.publish_rz = self.create_publisher(Float64, "/rz",10)

    def sensor_callback(self, msg):
        
        time1 = time.time()
        
        msg_send = Float64()
        msg_send.data = msg.data[0]
        self.publish_x.publish(msg_send);
        msg_send.data = msg.data[1]
        self.publish_y.publish(msg_send); msg_send.data = msg.data[2]
        self.publish_z.publish(msg_send); msg_send.data = msg.data[3]
        self.publish_rx.publish(msg_send); msg_send.data = msg.data[4]
        self.publish_ry.publish(msg_send); msg_send.data = msg.data[5]
        self.publish_rz.publish(msg_send);

        msg_send.data = msg.data[12]
        self.publish_fx.publish(msg_send); msg_send.data = msg.data[13]
        self.publish_fy.publish(msg_send); msg_send.data = msg.data[14]
        self.publish_fz.publish(msg_send); msg_send.data = msg.data[15]
        self.publish_mx.publish(msg_send); msg_send.data = msg.data[16]
        self.publish_my.publish(msg_send); msg_send.data = msg.data[17]
        self.publish_mz.publish(msg_send);

        time2 = time.time()
        print(f"\t - Time for publish task: {time2-time1}")

        
def main(args=None):
    rclpy.init(args=args)
    pub = plt_class()
    rclpy.spin(pub)
    pub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()