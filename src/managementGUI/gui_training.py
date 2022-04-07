import numpy as np
import sys
from datetime import datetime
import threading
import time

import pandas as pd
import PySimpleGUI as sg

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool, String, Empty
from custom_msg_srv.srv import SensorCall

q_keys = ["-q1-", "-q2-", "-q3-", "-q4-", "-q5-", "-q6-"]

q_home = [1.0, -1.0, 1.0, -1.505, -1.505, 0.0]

dataArray = Float32MultiArray

class ControlWindow():
    def __init__(self, dt):
        self.layoutTraining = [
            [
                sg.Text(text="Training iteration", key="train_status"),
                sg.Button(button_text="START", key="-training-")
            ]
        ]
        self.layoutGUI = [
            [
                sg.Button(button_text="Return to Home", key="-home-")
            ],
            [
                sg.Text(text="q1"),
                sg.In(default_text="0", key="-q1-")
            ],
            [
                sg.Text(text="q2"),
                sg.In(default_text="0", key="-q2-")
            ],
            [
                sg.Text(text="q3"),
                sg.In(default_text="0", key="-q3-")
            ],
            [
                sg.Text(text="q4"),
                sg.In(default_text="0", key="-q4-")
            ],
            [
                sg.Text(text="q5"),
                sg.In(default_text="0", key="-q5-")
            ],
            [
                sg.Text(text="q6"),
                sg.In(default_text="0", key="-q6-")
            ],
            [
                sg.Button(button_text="Send", key="-send-")
            ]
        ]
        self.layout = [
            [sg.Column(self.layoutTraining, key="-COL_TRAINING-", visible = True)],
            [sg.Column(self.layoutGUI, key="-COL_GUI-", visible = True)],
        ]

        self.window = sg.Window('Window Title', self.layout)

        # ROS setup
        self.node = rclpy.create_node("window_robot_scheduling")

        # Service setup
        self.ur_sensor_response = self.node.create_client(SensorCall, '/ur/sensor')
        # while not self.ur_sensor_response.wait_for_service(timeout_sec=1.0):
        #     self.node.get_logger().info('Real robot interface not available...')
        
        # publisher setup
        self.gui_publisher = self.node.create_publisher(dataArray, '/gui/position',1)
        self.teach_publisher = self.node.create_publisher(Bool, '/gui/teach',1)
        self.request_publisher = self.node.create_publisher(Empty, '/gui/request',1)

        # subscriber setup
        self.sensor_subscriber = self.node.create_subscription(Float32MultiArray, '/ur/data', self.data_callback, 1)

        # Data logging
        self.data_log = False
        self.iter = 0
        self.lock = threading.Lock()
        self.data = []
        self.stop = False
        self.bmsg = Bool()
        self.request_received = True
        # self.req = SensorCall.Request()

    def run(self):
        threadLoop = threading.Thread(target=self.dataLoop) 
        self.windowLoop()


    def windowLoop(self):
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED or event == 'Cancel':
                self.deactivateTeach()
                self.returnHome()

                self.lock.acquire()
                self.stop = True
                self.lock.release()
                break

            if event == "-send-":
                if not self.data_log:
                    self.sendWrittenQ(values=values)

            if event == "-home-":
                if not self.data_log:
                    self.returnHome()

            if event == "-training-":
                if self.data_log == True:
                    self.commuteLog()
                    self.updateIter()
                    self.window["-training-"].update("START")
                    self.deactivateTeach()
                    self.returnHome()
                elif self.data_log == False:
                    self.commuteLog()
                    self.window["-training-"].update("STOP")
                    self.activateTeach()
                    self.returnHome()

    def dataLoop(self):
        while True:
            if self.stop:
                name_time = datetime.now().strftime("%H_%M_%S")
                np.save(f"{name_time}.npy", self.data)
                break
            if self.data_log and self.request_received:
                self.request_publisher.publish(self.bmsg)
                self.request_received = False
    

    def data_callback(self, msg):
        if self.data_log:
            self.ur_sensor_response.request()
            self.data.append(msg.data + [time.time(), self.iter])
            self.request_received = True

    def returnHome(self):
        msg = Float32MultiArray()
        msg.data = q_home
        self.gui_publisher.publish(msg)
    
    def activateTeach(self):
        msg = Bool()
        msg.data = True
        self.teach_publisher.publish(msg)
    def deactivateTeach(self):
        msg = Bool()
        msg.data = False
        self.teach_publisher.publish(msg)

    def sendWrittenQ(self, values):
        msg = Float32MultiArray()
        msg.data = [float(values['-q1-']), float(values['-q2-']), float(values['-q3-']), float(values['-q4-']), float(values['-q5-']), float(values['-q6-'])]

    def commuteLog(self):
        self.lock.acquire()
        self.data_log = not self.data_log
        self.lock.release()

    def updateIter(self):
        self.iter += 1

    # def sendRequest(self):
    #     self.future = self.ur_sensor_response().call_async(self.req)
    #     while not self.future.is_done()

        

def main(args=None):
    rclpy.init()    
    window = ControlWindow(dt = float(sys.argv[1]))
    window.run()
    window.window.close()

if __name__=="__main__":
    main(sys.argv)