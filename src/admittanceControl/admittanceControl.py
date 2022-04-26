import numpy as np
import math as m

from utilities import separate_rotation_translation, ZYZ_conver, get_quaternion_from_euler


class AdmittanceControl:
    def __init__(self, mass_matrix, k_matrix, damp_matrix, desired_position, initial_position, orientation_rep=""):
        self.inv_mass = np.linalg.inv(mass_matrix)
        self.mass_matrix = mass_matrix
        self.k_matrix = k_matrix
        self.damp_matrix = damp_matrix

        self.desired_orientation, self.desired_position = separate_rotation_translation(desired_position)
        self.initial_orientation, self.initial_position = separate_rotation_translation(initial_position)

        if orientation_rep:
            self.desired_orientation = self.__orientation_transform__(self.desired_orientation, orientation_rep)
            self.initial_orientation = self.__orientation_transform__(self.initial_orientation, orientation_rep)

        self.actual_position = self.initial_position
        self.actual_orientation = self.initial_orientation

        self.actual_pos_orient = self.concatenate_vectors_conv_matrix(self.actual_position, self.actual_orientation)
        self.desired_pos_orient = self.concatenate_vectors_conv_matrix(self.desired_position, self.desired_orientation)

        self.actual_speed = np.array([[0]] * 6)
        self.actual_acceleration = np.array([[0]] * 6)

        self.orientation_rep = orientation_rep
        self.stored_actual_space_config = []
        self.stored_desired_space_config = []

    def step(self, dt, ex_force, store=False, desired_position=None):
        if desired_position is not None:
            self.desired_position, self.desired_orientation = separate_rotation_translation(desired_position)

            self.desired_pos_orient = self.concatenate_vectors_conv_matrix(self.desired_position,
                                                                           self.desired_orientation)

        error = self.__compute_error__()

        if self.orientation_rep == "euler":
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

            self.actual_acceleration = self.inv_mass * (np.dot(t_a.T, ex_force)
                                                        - self.damp_matrix * self.actual_speed
                                                        - self.k_matrix * error)

            self.actual_speed = self.actual_speed + self.actual_acceleration * dt
            self.actual_pos_orient = self.actual_pos_orient + self.actual_speed * dt

            self.actual_orientation = self.actual_pos_orient.getA1()[3:]
            self.actual_position = self.actual_pos_orient.getA1()[:3]

            if store:
                self.stored_actual_space_config.append(self.actual_position)
                self.stored_desired_space_config.append(self.desired_pos_orient)

    def __compute_error__(self):
        error_rc = []

        if self.orientation_rep == "euler":
            pos_error = self.actual_position - self.desired_position
            orient_error = self.actual_orientation - self.desired_orientation

            error_rc = self.concatenate_vectors_conv_matrix(pos_error, orient_error)

        return error_rc
    
    def update_K(self, new_K):
        self.k_matrix = new_K
    
    def update_damp(self, new_damp):
        self.damp_matrix = new_damp

    def get_stored_values(self):
        return self.stored_actual_space_config, self.stored_desired_space_config

    def reset_stored_values(self):
        self.stored_actual_space_config = []
        self.stored_desired_space_config = []

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
