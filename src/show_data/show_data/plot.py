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
        # preparing the plot
        #plt.ion()
        #self.i = 0
        #self.entire_data = np.zeros((max_samples,37))
        #self.fig, self.axs = plt.subplots(nrows=3, ncols=6)
        #self.axs[0,0].set_ylabel("x_pos"); self.axs[0,1].set_ylabel("y_pos"); self.axs[0,2].set_ylabel("z_pos"); self.axs[0,3].set_ylabel("x_orientation"); self.axs[0,4].set_ylabel("y_orientation"); self.axs[0,5].set_ylabel("z_orientation")
        #self.axs[1,0].set_ylabel("x_vel"); self.axs[1,1].set_ylabel("y_vel"); self.axs[1,2].set_ylabel("z_vel"); self.axs[1,3].set_ylabel("w_x"); self.axs[1,4].set_ylabel("w_y"); self.axs[1,5].set_ylabel("w_z")
        #self.axs[2,0].set_ylabel("f_x"); self.axs[2,1].set_ylabel("f_y"); self.axs[2,2].set_ylabel("f_z"); self.axs[2,3].set_ylabel("m_x"); self.axs[2,4].set_ylabel("m_y"); self.axs[2,5].set_ylabel("m_z")
        #self.updatePlot()
        #plt.show()

    def sensor_callback(self, msg):
        msg_send = Float64()
        time1 = time.time()
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

        #self.updateData(msg.data)
        time2 = time.time()
        print(f"\t - Time for publish task: {time2-time1}")

        #self.updatePlot()

    def updatePlot(self):
        print(f"Data length is: {self.entire_data.shape}")
        
        #update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,0]}
        #self.axs[0,0].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,1]};
        #self.axs[0,1].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,2]};
        #self.axs[0,2].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,3]};
        #self.axs[0,3].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,4]};
        #self.axs[0,4].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,5]};
        #self.axs[0,5].update(update);
        
        #update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,6]};
        #self.axs[1,0].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,7]};
        #self.axs[1,1].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,8]};
        #self.axs[1,2].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,9]};
        #self.axs[1,3].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,10]};
        #self.axs[1,4].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,11]};
        #self.axs[1,5].update(update)

        update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,12]};
        self.axs[2,0].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,13]};
        self.axs[2,1].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,14]};
        self.axs[2,2].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,15]};
        self.axs[2,3].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,16]};
        self.axs[2,4].update(update); update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,17]};
        self.axs[2,5].update(update);

        #self.axs[2,0].plot(self.entire_data[:,36], self.entire_data[:,12]); #update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,13]};
        #self.axs[2,1].plot(self.entire_data[:,36], self.entire_data[:,13]); #update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,14]};
        #self.axs[2,2].plot(self.entire_data[:,36], self.entire_data[:,14]); #update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,15]};
        #self.axs[2,3].plot(self.entire_data[:,36], self.entire_data[:,15]); #update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,16]};
        #self.axs[2,4].plot(self.entire_data[:,36], self.entire_data[:,16]); #update = {'xticks': self.entire_data[:,36], 'yticks': self.entire_data[:,17]};
        #self.axs[2,5].plot(self.entire_data[:,36], self.entire_data[:,17])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.draw()

    def updateData(self, data):
        print("In here")
        if self.i < max_samples:
            self.entire_data[self.i,:] = np.reshape(np.array(data), (1,37))
            self.i += 1
        else:
            for i in range(max_samples-1):
                self.entire_data[i,:] = self.entire_data[i+1,:]
            self.entire_data[max_samples-1,:] = np.reshape(np.array(data), (1,37))
        
def main(args=None):
    rclpy.init(args=args)
    pub = plt_class()
    rclpy.spin(pub)
    pub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()