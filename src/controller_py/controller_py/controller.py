import sys
import os
from cv2 import QT_FONT_LIGHT
import numpy as np
import pickle
import sklearn
import gmr

demo = 1

start_pos = 19
end_pos = 22

start_vel = 25

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, String, Empty

sys.path.append("src/admittanceControl")
import admittanceControl
sys.path.append("src/gmr")
from gaussian_mixture_regression import load_GMM, predict_GMR

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
        self.declare_parameter("mode")
        self.trajectory_path = self.get_parameter("trajectory").get_parameter_value().string_value
        self.dt = self.get_parameter("dt").get_parameter_value().double_value
        self.mode = self.get_parameter("mode").get_parameter_value().double_value
        # self.model_path = self.get_parameter("model").get_parameter_value().string_value
        
        if self.dt == '':
            print(f"[ERROR] Incorrect dt input: ({self.dt})")
            sys.exit()
        if self.trajectory_path == '':
            print(f"[ERROR] Incorrect dataset input: ({self.trajectory_path})")
            sys.exit()

        if self.mode == '':
            print(f"[ERROR] Incorrect dataset input: ({self.mode})")
            sys.exit()
        elif (self.mode != "v") and (self.mode != "p"):
            print(f"[ERROR] Incorrect executing mode: ({self.mode}). Only velocity(v) or position(p) are accepted")
            sys.exit()


        # if self.model_path == '':
        #     print(f"[ERROR] Incorrect dataset input: ({self.model_path})")
        #     sys.exit()

        # Load the model
        #try:
        #    self.model = load_GMM(filepath="")
        #except:
        #    print("[ERROR] ERROR when loading model.")
        #    sys.exit()
        
        # Load trajectory
        self.trajectory = np.load(self.trajectory_path)
        self.iter = 0
        print(f"First value is: {self.trajectory[demo,start_pos:end_pos,0]}")
        self.ref_pos = self.trajectory[demo,start_pos:end_pos,self.iter]
        self.ref_vel = self.trajectory[demo,start_vel:,self.iter]
        self.home = np.copy(self.ref)
        
        # Create controller instance
        #Kp, Dp = self.model.predict(np.zeros((3,1)))
        Mi = np.eye(3)
        Kv = np.eye(3)
        Kp = np.eye(3)
        self.controller = admittanceControl.AdmittanceControl(mass_matrix=Mi, k_matrix = Kp, damp_matrix=Kv, desired_position = self.trajectory[0,:3], initial_position = self.trajectory[0,:3], only_position=True, orientation_rep="")
        self.controller.load_parameter_matrix(Mi, Kp, Kv)

        # reset fields
        self.controller_active = False
        self.endPoint_achieved = False

        # Declare subscribers
        self.reset_subscriber = self.create_subscription(Bool, '/ur/reset', self.reset_callback, 10)
        self.sensor_subscriber = self.create_subscription(dataArray, "/sensor_data", self.sensor_callback, 10)

        # Declare publishers
        self.position_publisher = self.create_publisher(dataArray, '/ur/controller_position', 10)
        self.speed_publisher = self.create_publisher(dataArray, '/ur/controller_velocity', 10)
        # Go Home
        self.home_achieved = False
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
    
    def sensor_callback(self, msg):
        '''
        Update controller output based on the last sensor reading.
        '''
        point = np.array(msg.data[0:3])
        print(f"Check if the home is achieved or not.")
        if self.home_achieved == False:
            print(f"Distance to home is: {np.linalg.norm(self.home[:3]-point[:3])}")
            if np.linalg.norm(self.home-point) < 0.3:
                print(f"Achieved")
                self.home_achieved = True
        else:
            self.f = np.array(msg.data[12:15])
            #Kp, Kv, Mi = self.model.predict(self.f)
            #self.controller.load_parameter_matrix(Mi, Kp, Kv)

            #new_pos = self.controller.step(self.dt, self.f, False, self.ref.reshape((1,3)))
            #new_pos, new_speed = self.controller.step(self.dt, self.f, False, self.ref_pos.reshape((1,3)), self.ref_vel.reshape((1,3)))

            new_msg = dataArray()

            if self.mode == "p":
                new_msg.data = self.ref_pos.tolist() + list(msg.data[3:6])
                self.position_publisher.publish(new_msg)

            if self.mode == "v":
                new_msg.data = self.ref_vel.tolist() + [0, 0, 0]
                self.speed_publisher.publish(new_msg)

    def update_ref(self):
        '''
        Update reference trajectory point.
        '''
        if self.home_achieved:
            print("YES!")
            if not self.endPoint_achieved:
                if self.iter - 1< self.trajectory.shape[2]:
                    self.iter += 1
                    self.ref_pos = self.trajectory[demo,start_pos:end_pos, self.iter]
                    self.ref_vel = self.trajectory[demo,start_vel:, self.iter]
                else:
                    self.endPoint_achieved = True
                    self.get_logger().info("Last point set as ref point.")
        
        else:
            print(f"Going to home: {self.home}")
            self.goHome()
    
    def goHome(self):
        msg = dataArray()
        msg.data = self.home.tolist() + [1.55, 2.71, 0.035]
        msg.data[0] += 0.18
        msg.data[1] += 0.18
        msg.data[2] /= 2
        self.position_publisher.publish(msg)
        print("Trying to go home...")


def main(args=None):
    '''
    Main function that initializes the node.
    '''
    rclpy.init(args=args)

    controller = controllerNode()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    
    main()