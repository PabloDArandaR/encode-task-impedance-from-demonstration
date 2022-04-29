import sys
import os
from cv2 import QT_FONT_LIGHT
import numpy as np
import pickle
import sklearn
import gmr

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, String, Empty

sys.path.append("src/admittanceControl")
from admittanceControl import admittanceControl
sys.path.append("src/MANUsEmbeddingOfTheModel")
from manu import manuModel

q_home = [-0.202008, -0.18495, 0.4007, 1.929, 2.3557, 0.012822]

dataArray = Float64MultiArray

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

        # Declare parameters
        self.declare_parameter("trajectory")
        self.declare_parameter("dt")
        self.trajectory_path = self.get_parameter("trajectory").get_parameter_value().string_value
        self.dt = self.get_parameter("dt").get_parameter_value().double_value
        self.model_path = self.get_parameter("model").get_parameter_value().string_value
        
        if self.dt == '':
            print(f"[ERROR] Incorrect dt input: ({self.dt})")
            sys.exit()
        if self.trajectory_path == '':
            print(f"[ERROR] Incorrect dataset input: ({self.trajectory_path})")
            sys.exit()
        if self.model_path == '':
            print(f"[ERROR] Incorrect dataset input: ({self.model_path})")
            sys.exit()

        # Load the model
        try:
            self.model = manuModel(self.model_path)
        except:
            print("[ERROR] ERROR when loading model.")
            sys.exit()
        
        # Load trajectory
        self.trajectory = np.load(self.trajectory_path)
        
        # Create controller instance
        Kp, Dp = self.model.predict(np.zeros((3,1)))
        self.controller = admittanceControl.AdmittanceControl(mass_matrix=np.eye(3), k_matrix = Kp, damp_matrix=Dp, desired_position = self.trajectory[0,:3], desired_position = self.trajectory[0,:3], orientation_rep="")

        # reset fields
        self.controller_active = False
        self.endPoint_achieved = False

        # Declare subscribers
        self.reset_subscriber = self.create_subscription(Bool, '/ur/reset', self.reset_callback, 10)
        self.sensor_subscriber = self.create_subscription(dataArray, "/sensor_data", self.sensor_callback, 10)

        # Declare publishers
        self.output_publisher = self.create_publisher(dataArray, '/ur/output_controller', 10)
        # Go Home
        self.goHome()

        # Declare timer for update of reference position based on the input file
        self.ref_timer = self.create_timer(self.dt, self.update_ref)


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
        self.controller_active = False
        self.endPoint_achieved = False
    
    def calculateAction(self): # TODO: Based on the real controller 
        '''
        Calculate the action based on the parameters that have been received in the last messages.
        '''
        M, D, K = self.action_model
        A = self.q + self.q_1 + self.dq + self.dq_1 + self.tau

        return A
    
    def sensor_callback(self, msg):
        '''
        Update controller output based on the last sensor reading.
        '''
        f = np.array(msg.data[12:15])
        Kp, Kv, Im = self.model.predict(f)
        self.controller.update_K(Kp)
        self.controller.update_damp(Kv)
    
    def updateState(self, data):
        '''
        Update inner state based on the information received from the sensor.
        '''
        if self.first:
            self.q = data[:6]
            self.dq = data[6:12]
            self.tau = data[12:18]
            self.q_1 = data[:6]
            self.dq_1 = data[6:12]
            self.tau_1 = data[12:18]
            self.first = False
        else:
            self.q_1 = self.q
            self.dq_1 = self.dq
            self.tau_1 = self.tau
            self.q = data[:6]
            self.dq = data[6:12]
            self.tau = data[12:18]

    def update_ref(self):
        '''
        Update reference trajectory point.
        TODO add how the matrix are also read and the different parameters for the controller
        '''
        if self.controller_active and not self.endPoint_achieved:
            new_el = next(self.trajectory_iter)
            if new_el != None:
                self.ref = new_el
            else:
                self.endPoint_achieved = True
                self.get_logger().info("Last point set as ref point.")
    
    def goHome(self):
        msg = dataArray()
        msg.data = q_home
        self.output_publisher.publish(msg)


def main(args=None):
    '''
    Main function that initializes the node.
    '''
    rclpy.init(args=args)

    controller = controllerNode(0.1,"")
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    
    main()