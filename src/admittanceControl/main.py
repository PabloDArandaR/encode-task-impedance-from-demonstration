from admittanceControl import *
import matplotlib.pyplot as plt

# the parameters are taken using the page (UR5e)
# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
DH_Parameters_a = [0, -0.425, -0.3922, 0, 0, 0]
DH_Parameters_d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
DH_Parameters_alpha = [math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0]


# Parameters for the 2nd order spring, they can be computed as a matrix or int, float
# float/int ex mp = 1
# matrix ex: mp = [ [1, 0, 0], [0, 1, 0], [0, 0, 1]]. It must be a 3x3 matrix.
mp = 1
kp = 10
mo = 0.1
ko = 1.5

# Initial config
q_i = [0, 0, 0, 0, 0, 0]

# Desired config
q_d = [1, math.pi/3, math.pi/3, 0, 0, 0]

# Parameters of the simulation:
tmax = 25
dt = 1 / 10

# Change in case of manipulating the orientation as well.
only_position = True


def show_stored_values(stored_values, time_range, save_plot=False, name_file=""):
    x_actual_pos = []
    y_actual_pos = []
    z_actual_pos = []

    for pos in stored_values:
        x_actual_pos.append(pos[0])
        y_actual_pos.append(pos[1])
        z_actual_pos.append(pos[2])

    figure, axis = plt.subplots(3, 1)

    axis[0].plot(time_range, x_actual_pos)
    axis[0].set_title("X-axis actual position")

    axis[1].plot(time_range, y_actual_pos)
    axis[1].set_title("Y-axis actual position")

    axis[2].plot(time_range, z_actual_pos)
    axis[2].set_title("Z-axis actual position")

    figure.tight_layout()

    if save_plot:
        plt.savefig(name_file)
    else:
        plt.show()


def main():

    # return vector of individual transformations of each joint and the combined transformation
    _, comb_trans_i = comp_DH_trans(q_i, DH_Parameters_a, DH_Parameters_d, DH_Parameters_alpha)
    _, comb_trans_d = comp_DH_trans(q_d, DH_Parameters_a, DH_Parameters_d, DH_Parameters_alpha)

    # The last transformation is chosen (TCP)
    desired_pos = comb_trans_d[5]
    initial_pos = comb_trans_i[5]

    # if we are going to use only the positions, the M matrix, K matrix and V matrix should be 3x3 otherwise 6x6
    if only_position:
        ma_aux = mp
        ka_aux = kp

        # In case they are float or int, they are converted to a matrix
        if type(ma_aux) is int or type(ma_aux) is float:
            ma_aux = [
                [mp, 0, 0],
                [0, mp, 0],
                [0, 0, mp]
            ]
            ka_aux = [
                [kp, 0, 0],
                [0, kp, 0],
                [0, 0, kp]
            ]

        da_aux = critical_damping_formula(ma_aux, ka_aux)

        # For taking only the translation part
        _, desired_pos = separate_rotation_translation(desired_pos)
        _, initial_pos = separate_rotation_translation(initial_pos)

        # Convert vector to array 3x1
        desired_pos = np.array(desired_pos)[np.newaxis].T
        initial_pos = np.array(initial_pos)[np.newaxis].T

    else:
        ma_aux, ka_aux, da_aux = compute_parameters_matrix(mo, ko, mp, kp)

    admittance_control = AdmittanceControl(ma_aux, ka_aux, da_aux, desired_pos, initial_pos, only_position, "euler")

    time_range = np.arange(0.0, tmax, dt)
    for i in time_range:

        # If we are going to compute only the positions, we need a 3x1 matrix otherwise 6x1.
        # In this case we are going to simulate an external force.
        if not only_position:
            if 5 <= i < 10:
                tau = [[1], [2], [3], [0], [0], [0]]
            elif 15 <= i < 20:
                tau = [[0], [0], [0], [1], [0.5], [1]]
            else:
                tau = [[0]] * 6
        else:
            if 5 <= i < 10:
                tau = [[1], [2], [3]]
            else:
                tau = [[0]] * 3

        admittance_control.step(dt, np.array(tau), True)

        # In case of changing the desired pose/position:
        # admittance_control.step(dt, np.array(tau), True, desired_pose)

    stored_actual_pos, _ = admittance_control.get_stored_values()

    show_stored_values(stored_actual_pos, time_range)


if __name__ == '__main__':
    main()
