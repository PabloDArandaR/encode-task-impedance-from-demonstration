import numpy as np
import sys
import os
from datetime import datetime
import threading
import time

import pandas as pd
import PySimpleGUI as sg

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool, String, Empty

q_keys = ["-q1-", "-q2-", "-q3-", "-q4-", "-q5-", "-q6-"]

q_home = [0.2, -2.2, 2.2, -1.5, -1.5, 0.0]

dataArray = Float32MultiArray

class ControlWindow():
    def __init__(self, dt, task):
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
        
        # publisher setup
        self.gui_publisher = self.node.create_publisher(dataArray, '/gui/position',10)
        self.teach_publisher = self.node.create_publisher(Bool, '/gui/teach',10)

        # subscriber setup
        self.sensor_subscriber = self.node.create_subscription(dataArray, "/sensor_data", self.response_callback, 10)

        # Data logging
        self.data_log = False
        self.iter = 0
        self.lock = threading.Lock()
        self.data = []
        self.dt = dt
        self.task = task

        # Check folders exist and/or create them
        if not os.path.isdir(f"resources/training_data/task_{self.task}"):
            os.mkdir(f"resources/training_data/task_{self.task}")

        # Check number of iterations already done for the given task
        if len(os.listdir(f"resources/training_data/task_{self.task}")):
            n_iters = [int(val[val.find("_")+1:-4]) for val in os.listdir(f"resources/training_data/task_{self.task}") if val[-4:] == ".npy"]
            self.iter = max(n_iters) + 1

    #############################################################################################################
    # DATA RELATED FUNCTIONS

    def response_callback(self,msg):
        if self.data_log:
            self.lock.acquire()
            data = list(msg.data)
            self.data.append(data)
            self.lock.release()

    #############################################################################################################
    # WINDOW RELATED FUNCTIONS

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
        self.gui_publisher.publish(msg)

    def commuteLog(self):
        self.lock.acquire()
        self.data_log = not self.data_log
        self.lock.release()

    def updateIter(self):
        self.iter += 1
    
    def storeData(self):
        self.lock.acquire()
        np.save(f"resources/training_data/task_{self.task}/{self.iter}.npy", self.data)
        self.data = []
        self.lock.release()

    #############################################################################################################
    # RUN RELATED FUNCTIONS
    
    def windowLoop(self):
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED or event == 'Cancel':
                self.deactivateTeach()
                self.returnHome()
                break

            if event == "-send-":
                if not self.data_log:
                    self.sendWrittenQ(values=values)

            if event == "-home-":
                if not self.data_log:
                    self.returnHome()

            if event == "-training-":
                if self.data_log == True:
                    print("FINISHING TRAINING")
                    self.commuteLog()
                    self.window["-training-"].update("START")
                    self.deactivateTeach()
                    self.returnHome()
                    self.storeData()
                    self.updateIter()

                elif self.data_log == False:
                    print(f"STARTING TRAINING {self.iter}")
                    self.commuteLog()
                    self.window["-training-"].update("STOP")
                    self.returnHome()
                    self.activateTeach()

    def dataLoop(self):
        rclpy.spin(self.node)
        self.node.destroy_node()
    
    def run(self):
        threadLoop = threading.Thread(target=self.dataLoop) 
        threadLoop.start()
        self.windowLoop()

        

def main(args=None):
    rclpy.init(args=args)    
    window = ControlWindow(dt = float(sys.argv[1]), task=sys.argv[2])
    window.run()
    window.window.close()
    rclpy.shutdown()

if __name__=="__main__":
    main(sys.argv)