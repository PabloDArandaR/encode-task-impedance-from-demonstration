from utilities import *
import math as m


class AdmittanceControl:
    """ This class represents the admittance control. In order to use it, the parameters for the second order spring
        have to be entered as well the initial position and the desired position. The method step, simulate a step
        in the admittance control. Each step can be stored. The method "get_stored_values" gives the access to these
        stored values. The method "load_parameter_matrix" allows to change the parameters of the second order spring.

        In its constructor:
        Parameters:
        mass_matrix (np.ndarray): Mass matrix of the 2nd order spring.
        k_matrix (np.ndarray): K matrix of the 2nd order spring.
        damp_matrix (np.ndarray): Damping matrix of the 2nd order spring.
        desired_position (np.ndarray): Desired pose/position.
        initial_position (np.ndarray): Intial pose/position.
        only_position (bool): Bool to inform whether the admittance control is going to work with the position or not.
        orientation_rep (string): The orientation representation. It can be "euler" or "quaternion"

    """
    def __init__(self, mass_matrix, k_matrix, damp_matrix, desired_position, initial_position, only_position=False,
                 orientation_rep=""):

        self.only_position = only_position
        self.inv_mass = np.linalg.inv(mass_matrix)
        self.mass_matrix = mass_matrix
        self.k_matrix = k_matrix
        self.damp_matrix = damp_matrix

        if only_position:
            self.desired_orientation = ZERO_DEGREES_TRANS
            self.desired_position = desired_position

            self.initial_orientation = ZERO_DEGREES_TRANS
            self.initial_position = initial_position

            self.actual_position = self.initial_position
            self.orientation_rep = ""
        else:
            self.desired_orientation, self.desired_position = separate_rotation_translation(desired_position)
            self.initial_orientation, self.initial_position = separate_rotation_translation(initial_position)

            if orientation_rep:
                self.desired_orientation = self.__orientation_transform__(self.desired_orientation, orientation_rep)
                self.initial_orientation = self.__orientation_transform__(self.initial_orientation, orientation_rep)

            self.actual_orientation = self.initial_orientation
            self.actual_position = self.initial_position

            self.actual_pos_orient = self.concatenate_vectors_conv_matrix(self.actual_position, self.actual_orientation)
            self.desired_pos_orient = self.concatenate_vectors_conv_matrix(self.desired_position,
                                                                           self.desired_orientation)

            self.orientation_rep = orientation_rep

        self.actual_speed = np.array([[0]] * (3 if only_position else 6))
        self.actual_acceleration = np.array([[0]] * (3 if only_position else 6))

        self.stored_actual_position = []
        self.stored_desired_position = []

    def step(self, dt, ex_force, store=False, desired_pose=None):
        """ Simulate one step further of the Admittance control exposed to an external force. In every step, the desired
            pose/position can be changed.

            Parameters:
            dt (float): The difference of time between the previous step and this one.
            ex_force (np.ndarray): the external force
            store (bool): indicates whether this step should be stored or not.
            desired_pose (np.ndarray): The new desired pose/position.

            Returns:
            np.ndarray: Actual position.
            np.ndarray: Actual orientation, in case of having only the position this space will be None

           """

        if self.only_position:
            if desired_pose is not None:
                self.desired_position = desired_pose
                self.desired_pos_orient = self.concatenate_vectors_conv_matrix(self.desired_position,
                                                                               self.desired_orientation)
        else:
            if desired_pose is not None:
                self.desired_position, self.desired_orientation = separate_rotation_translation(desired_pose)

                self.desired_pos_orient = self.concatenate_vectors_conv_matrix(self.desired_position,
                                                                               self.desired_orientation)

        error = self.__compute_error__()

        k_dot_err = np.dot(self.k_matrix, error)
        damp_dot_speed = np.dot(self.damp_matrix, self.actual_speed)

        if self.orientation_rep == "euler" or self.only_position:

            if not self.only_position:
                c1 = m.cos(self.actual_orientation[0])
                s1 = m.sin(self.actual_orientation[0])

                c2 = m.cos(self.actual_orientation[1])
                s2 = m.sin(self.actual_orientation[1])

                t_a = np.array(
                    [
                        [1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],

                        [0, 0, 0, 0, -s1, c1 * s2],
                        [0, 0, 0, 0, c1, s1 * s2],
                        [0, 0, 0, 1, 0, c2],
                    ]
                )

                self.actual_acceleration = self.inv_mass * (np.dot(t_a.T, ex_force) - damp_dot_speed - k_dot_err)
            else:
                self.actual_acceleration = np.dot(self.inv_mass, (ex_force - damp_dot_speed - k_dot_err))

            self.actual_speed = self.actual_speed + self.actual_acceleration * dt

            if not self.only_position:
                self.actual_pos_orient = self.actual_pos_orient + self.actual_speed * dt

                self.actual_orientation = self.actual_pos_orient.getA1()[3:]
                self.actual_position = self.actual_pos_orient.getA1()[:3]

            else:
                self.actual_position = self.actual_position + self.actual_speed * dt

            if store:
                self.stored_actual_position.append(self.actual_position)
                self.stored_desired_position.append(self.desired_position)

        return [self.actual_position, None if self.only_position else self.actual_orientation]

    def get_stored_values(self):
        """ Get the stored values from the executed steps.

            Returns:
            vector of np.ndarray: Stored actual position.
            vector of np.ndarray: Stored desired position

           """
        return [self.stored_actual_position, self.stored_desired_position]

    def reset_stored_values(self):
        """ Reset the stored values from the executed steps.

           """
        self.stored_actual_position = []
        self.stored_desired_position = []

    def load_parameter_matrix(self, mi, kpi, kvi=None):
        """ Load new K matrix, M matrix and damping matrix. In case of not having a new Kvi Matrix, the critical damping
            will be computed from the new K matrix and M matrix.

            Parameters:
            mi (np.ndarray): The M matrix.
            kpi (np.ndarray): The K matrix.
            kvi (np.ndarray): The damping matrix


           """

        if None is kvi:
            critical_damping_formula(mi, kpi)

        mi = check_transform_np_array(mi)
        kpi = check_transform_np_array(kpi)
        kvi = check_transform_np_array(kvi)

        if not self.only_position:
            mi = expand_matrix(mi, 6, 6)
            kpi = expand_matrix(kpi, 6, 6)
            kvi = expand_matrix(kvi, 6, 6)

        self.inv_mass = np.linalg.inv(mi)
        self.mass_matrix = mi
        self.k_matrix = kpi
        self.damp_matrix = kvi

    def update_K(self, new_K):
        self.k_matrix = new_K

    def update_damp(self, new_damp):
        self.damp_matrix = new_damp

    def __compute_error__(self):
        error_rc = []

        if not self.only_position:
            if self.orientation_rep == "euler":
                pos_error = self.actual_position - self.desired_position
                orient_error = self.actual_orientation - self.desired_orientation

                error_rc = self.concatenate_vectors_conv_matrix(pos_error, orient_error)
        else:
            error_rc = self.actual_position - self.desired_position
            error_rc = np.array(error_rc)

        return error_rc

    @staticmethod
    def concatenate_vectors_conv_matrix(vect1, vect2):
        return np.concatenate([vect1, vect2])[np.newaxis].T

    @staticmethod
    def __orientation_transform__(rotation_matrix, orientation_rep):
        rv = []

        if orientation_rep == "euler" or orientation_rep == "quaternion":
            rv = ZYZ_conver(rotation_matrix, True)

        if orientation_rep == "quaternion":
            rv = get_quaternion_from_euler(rv)

        return np.array(rv) if rv else []
