import sys
import os
import numpy as np
import time

sys.path.append("src")

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, String, Empty, Float64

sys.path.append("src/admittanceControl")
import admittanceControl
sys.path.append("src/gmr")
from gaussian_mixture_regression import load_GMM, predict_GMR

demo = 0

start_pos = 19
end_pos = 22
start_Kp = 1
end_Kp = 7
start_Kv = 7
end_Kv = 13
start_Im = 13
end_Im = 19

start_vel = 25

q_home = [-0.202008, -0.18495, 0.4007, 1.929, 2.3557, 0.012822]

dataArray = Float64MultiArray

def read3x3Matrix(input: np.array):
    output = np.zeros((3,3))
    output[0,0] =input[0]
    output[1,1] =input[3]
    output[2,2] =input[5]

    output[0,1] =input[1]
    output[1,0] =input[1]

    output[0,2] =input[2]
    output[2,0] =input[2]


    output[1,2] =input[4]
    output[2,1] =input[4]
    

    return output

class controllerNode(Node):
    '''
    Controller node that handles the messages received from the robot and
    simulation and sends the required actions/speed based on the parameters
    obtained from the trained model.

    Attributes
    ----------
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
        self.declare_parameter("model")
        self.trajectory_path = self.get_parameter("trajectory").get_parameter_value().string_value
        self.dt = self.get_parameter("dt").get_parameter_value().double_value
        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.model = self.get_parameter("model").get_parameter_value().string_value
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

        if self.model == '':
            print(f"[WARNING] No model being used.")
            sys.exit()
        else:
            self.basename, self.modelpath = os.path.split(self.model)
            print(f"MODEL_PATH: {self.modelpath}")
            print(f"BASE_NAME: {self.basename}")
            self.force_predict_model = load_GMM(self.basename, self.modelpath)
        
        # Load trajectory
        self.trajectory = np.load(self.trajectory_path)
        self.iter = 0
        self.fz_1 = 0
        self.fz = 0
        self.counter = 0

        print(f"First value is: {self.trajectory[demo,start_pos:end_pos,0]}")
        self.ref_pos = self.trajectory[demo,start_pos:end_pos,self.iter]
        self.ref_vel = self.trajectory[demo,start_vel:,self.iter]

        # FORCE MODEL BASED PREDICTION
        if True:
            self.out = predict_GMR( gmm = self.force_predict_model, timestamp = np.float64(0))
            self.Kp = read3x3Matrix(self.out[:6])
            self.Kv = read3x3Matrix(self.out[6:12])
            self.Mi = read3x3Matrix(self.out[12:18])
            self.Kp[0,0] *= 10
            self.Kp[1,1] *= 10

            self.Kv[0,0] *= 10
            self.Kv[1,1] *= 10

            self.Mi[0,0] *= 10
            self.Mi[1,1] *= 10

        # TIME MODEL BASED PREDICTION
        if False:
            out = predict_GMR( gmm = self.force_predict_model, timestamp = np.array([0]))
            self.Kp = read3x3Matrix(out[:6])
            self.Kv = read3x3Matrix(out[6:12])
            self.Mi = read3x3Matrix(out[12:18])
            self.ref_pos = out[18:].reshape((3,1))
        # WHAT WE USE WITH NO MODEL
        if False:
            self.Kp = read3x3Matrix(self.trajectory[demo,start_Kp:end_Kp,self.iter])
            self.Mi = read3x3Matrix(self.trajectory[demo,start_Im:end_Im,self.iter])
            self.Kv = read3x3Matrix(self.trajectory[demo,start_Kv:end_Kv,self.iter])
            self.Kp[0,0] *= 10
            self.Kp[1,1] *= 10
            self.Kv[0,0] *= 10
            self.Kv[1,1] *= 10
            self.Mi[0,0] *= 10
            self.Mi[1,1] *= 10
        self.t_1 = time.time()
        self.t = time.time()

        self.home_pos = np.copy(self.ref_pos).tolist()
        self.home_angular_vel = [0.0,0.0,0.0]
        
        # Create controller instance
        self.controller = admittanceControl.AdmittanceControl(mass_matrix=self.Mi, k_matrix = self.Kp, damp_matrix=self.Kv, desired_position = self.ref_pos, initial_position = self.ref_pos, only_position=True, orientation_rep="")

        # reset fields
        self.controller_active = False
        self.endPoint_achieved = False

        # Declare subscribers
        self.reset_subscriber = self.create_subscription(Bool, '/ur/reset', self.reset_callback, 10)
        self.sensor_subscriber = self.create_subscription(dataArray, "/sensor_data", self.sensor_callback, 10)

        # Declare publishers
        self.position_publisher = self.create_publisher(dataArray, '/ur/controller_position', 10)
        self.speed_publisher = self.create_publisher(dataArray, '/ur/controller_velocity', 10)
        self.data_publisher = self.create_publisher(dataArray, '/data', 10)
        self.xref_publisher = self.create_publisher(Float64, '/xref', 10)
        self.yref_publisher = self.create_publisher(Float64, '/yref', 10)
        self.zref_publisher = self.create_publisher(Float64, '/zref', 10)

        # Go Home
        self.home_achieved = False
        self.first_received = False

        # Declare timer for update of reference position based on the input file
        self.ref_timer = self.create_timer(self.dt, self.update_ref)
        self.matrix_timer = self.create_timer(1.0, self.print_matrix)


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
    
    def print_matrix(self):
        print("#############################################################")
        print(f"Matrix_pre Kv: {self.Kv}")
        print(f"Matrix_pre Kp: {self.Kp}")
        print(f"Matrix_pre Mi: {self.Mi}")
        print(f"Force fz: {self.fz}")
        print(f"Converted force: {np.float64(self.fz)}")
        print(f"out: {self.out[5]}")


    def sensor_callback(self, msg):
        '''
        Update controller output based on the last sensor reading.
        '''

        # Setup home if it hasn't been setup yet
        if not self.first_received:
            self.home_orientation = list(msg.data[3:6])
            self.home = self.home_pos + self.home_orientation
            self.first_received = True
        
        point = np.array(msg.data[0:3])
        if self.home_achieved == False:
            self.goHome()
            if np.linalg.norm(self.home_pos-point) < 0.05:
                # self.get_logger().info(f"HOME Achieved")
                self.home_achieved = True
        else:
            self.f = 0.1*np.array(msg.data[12:15])
            self.new_position = np.array(msg.data[0:3])
            self.new_velocity = np.array(msg.data[6:9])
            self.t_1 = self.t
            self.t = time.time()
            
            # FORCE MODEL BASED PREDICTION
            if True:
                alpha = 0.1
                self.fz_1 = self.fz
                self.fz = self.fz_1*(1-alpha) + self.f[2]*alpha
                self.out = predict_GMR( gmm = self.force_predict_model, timestamp = np.float64(self.fz))
                self.Kp = read3x3Matrix(self.out[:6])
                self.Kv = read3x3Matrix(self.out[6:12])
                self.Mi = read3x3Matrix(self.out[12:18])
                
                self.Kp[0,0] *= 10
                self.Kp[1,1] *= 10

                self.Kv[0,0] *= 10
                self.Kv[1,1] *= 10
                #self.Kv[2,2] *= 100

                self.Mi[0,0] *= 10
                self.Mi[1,1] *= 10
                self.controller.load_parameter_matrix(self.Mi, self.Kp, self.Kv)
            
            new_pos, new_orientation, new_vel = self.controller.step(self.t-self.t_1, self.f, self.new_position, self.new_velocity, False, self.ref_pos)

            # SEND THE DATA
            new_msg = dataArray()

            if self.mode == "p":
                new_msg.data = new_pos.tolist() + self.home_orientation
                self.position_publisher.publish(new_msg)
            elif self.mode == "v":
                new_msg.data = new_vel.tolist() + self.home_angular_vel
                self.speed_publisher.publish(new_msg)
            
            
            new_msg = dataArray()
            new_msg.data = list(msg.data[:3]) + self.ref_pos.reshape((1,3)).tolist()[0] + list(msg.data[6:9]) + list(msg.data[12:15]) + self.Kp.reshape((1,9)).tolist()[0] + self.Kv.reshape((1,9)).tolist()[0] + self.Mi.reshape((1,9)).tolist()[0] + [msg.data[-1]]
            self.data_publisher.publish(new_msg)
            
            msg = Float64()
            msg.data = self.ref_pos[0]
            self.xref_publisher.publish(msg)
            msg.data = self.ref_pos[1]
            self.yref_publisher.publish(msg)
            msg.data = self.ref_pos[2]
            self.zref_publisher.publish(msg)

    def update_ref(self):
        '''
        Update reference trajectory point.
        '''
        if self.home_achieved:
            if not self.endPoint_achieved:
                if self.iter < self.trajectory.shape[2] - 1:
                    self.iter += 1
                    self.ref_pos = self.trajectory[demo,start_pos:end_pos, self.iter]
                    self.ref_vel = self.trajectory[demo,start_vel:, self.iter]
                    if False:
                        self.Kp = read3x3Matrix(self.trajectory[demo,start_Kp:end_Kp,self.iter])
                        self.Mi = read3x3Matrix(self.trajectory[demo,start_Im:end_Im,self.iter])
                        self.Kv = read3x3Matrix(self.trajectory[demo,start_Kv:end_Kv,self.iter])
                        self.Kp[0,0] *= 10
                        self.Kp[1,1] *= 10
                        self.Kv[0,0] *= 10
                        self.Kv[1,1] *= 10
                        self.Mi[0,0] *= 10
                        self.Mi[1,1] *= 10
                        self.controller.load_parameter_matrix(self.Mi, self.Kp, self.Kv)
                        print(f"Matrix Kv: {self.Kv[2,2]}")
                        print(f"Matrix Kp: {self.Kp[2,2]}")
                        print(f"Matrix Mi: {self.Mi[2,2]}")
                    
                else:
                    self.ref_vel = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
                    self.endPoint_achieved = True
                    self.get_logger().info("Last point set as ref point.")

        elif self.first_received:
            self.goHome()
    
    def goHome(self):
        msg = dataArray()
        #print(f"[INFO] Home position: {self.home_pos + self.home_orientation}")
        msg.data = self.home_pos + self.home_orientation + [0.05, 0.05]
        self.position_publisher.publish(msg)
        #print("Trying to go home...")


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