import math
import numpy as np

# Limit for decimals. A number under this limit will be taken as 0
limit_dec = math.pow(10, -15)
ZERO_DEGREES_TRANS = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])


def check_transform_np_array(val):
    if not (type(val) is np.ndarray):
        val = np.array(val)
    return val


def separate_rotation_translation(matrix):
    rotation, translation = [], []

    for i in range(3):
        rotation.append([])
        for p in matrix[i][0:3]:
            rotation[i].append(p)
        translation.append(matrix[i][3])

    return np.array(rotation), np.array(translation)


def ZYZ_conver(rotm, inv):
    eul1 = math.atan2(rotm.item(1, 2), rotm.item(0, 2))
    sp = math.sin(eul1)
    cp = math.cos(eul1)
    eul2 = math.atan2(cp * rotm.item(0, 2) + sp * rotm.item(1, 2), rotm.item(2, 2))
    eul3 = math.atan2(-sp * rotm.item(0, 0) + cp * rotm.item(1, 0), -sp * rotm.item(0, 1) + cp * rotm.item(1, 1))

    return [eul1, eul2, eul3] if not inv else [-1*eul1, -1*eul2, -1*eul3]


def get_quaternion_from_euler(angles):

    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]

    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def compute_6_by_6_diagonal_matrix_two_value(val1, val2):
    if type(val1) is int or type(val1) is float:
        aux_m = np.matrix([
            [val1, 0, 0, 0, 0, 0],
            [0, val1, 0, 0, 0, 0],
            [0, 0, val1, 0, 0, 0],
            [0, 0, 0, val2, 0, 0],
            [0, 0, 0, 0, val2, 0],
            [0, 0, 0, 0, 0, val2]
        ])
    else:
        if not (type(val1) is np.ndarray):
            val1 = np.array(val1)
            val2 = np.array(val2)

        zero_arr = np.array([[0] * 3] * 3)
        val1_z = np.concatenate((val1, zero_arr), axis=1)
        val2_z = np.concatenate((zero_arr, val2), axis=1)
        aux_m = np.concatenate((val1_z, val2_z))

    return aux_m


def trans(axis, a_t, theta_t):
    t_r = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    ci = math.cos(theta_t)
    si = math.sin(theta_t)

    if ci < limit_dec:
        ci = 0

    if si < limit_dec:
        si = 0

    if axis == 'x':
        t_r[0] = [1, 0, 0, a_t]
        t_r[1] = [0, ci, -si, 0]
        t_r[2] = [0, si, ci, 0]

    elif axis == 'y':
        t_r[0] = [ci, 0, si, 0]
        t_r[1] = [0, 1, 0, a_t]
        t_r[2] = [-si, 0, ci, 0]

    elif axis == 'z':
        t_r[0] = [ci, -si, 0, 0]
        t_r[1] = [si, ci, 0, 0]
        t_r[2] = [0, 0, 1, a_t]

    return t_r


def comp_DH_trans(joint_config, DH_Parameters_a, DH_Parameters_d, DH_Parameters_alpha):
    sing_trans = []
    comb_transf = []

    for i in range(len(DH_Parameters_a)):
        trans_z = np.array(trans('z', DH_Parameters_d[i], joint_config[i]))
        trans_x = np.array(trans('x', DH_Parameters_a[i], DH_Parameters_alpha[i]))

        a_t = np.dot(trans_z, trans_x)
        sing_trans.append(a_t)

        if i == 0:
            comb_transf.append(a_t)
        else:
            comb_transf.append(np.dot(comb_transf[i - 1], a_t))

    return sing_trans, comb_transf


def critical_damping_formula(m, k):
    """Compute the critical damping.

        Parameters:
        m (int/float/array/np.array): The mass.
        k (int/float/array/np.array): The k parameter.

        Returns:
        np.ndarray/float: The computed damping

       """
    if type(m) is int or type(m) is float:
        aux_d = 2*math.sqrt(m*(k+1))
    else:
        org_length = len(m)
        if not(type(m) is np.ndarray):
            m = np.array(m)
            k = np.array(k)

        aux_d = 2 * np.sqrt(m * (k + np.eye(org_length)))

    return aux_d


def compute_critical_damping(mo_f, ko_f, mp_f, kp_f):
    """Compute the damping parameter for the 2nd order spring system.

        Parameters:
        mo_f (int/float/array/np.array): The mass for the orientation.
        ko_f (int/float/array/np.array): The k parameter for the orientation.
        mp_f (int/float/array/np.array): The mass for the position.
        kp_f (int/float/array/np.array): The k parameter for the position.

        Returns:
        np.ndarray: The computed mass matrix 6x6
        np.ndarray: The computed k matrix 6x6
        np.ndarray: The computed damping matrix 6x6

       """
    aux_do = critical_damping_formula(mo_f, ko_f)
    aux_dp = critical_damping_formula(mp_f, kp_f)

    return aux_dp, aux_do


def compute_parameters_matrix(mo_f, ko_f, mp_f, kp_f):
    """Compute the damping parameter for the 2nd order spring system.

        Parameters:
        mo_f (int/float/array/np.array): The mass for the orientation.
        ko_f (int/float/array/np.array): The k parameter for the orientation.
        mp_f (int/float/array/np.array): The mass for the position.
        kp_f (int/float/array/np.array): The k parameter for the position.

        Returns:
        np.ndarray: The computed mass matrix 6x6
        np.ndarray: The computed k matrix 6x6
        np.ndarray: The computed damping matrix 6x6

       """

    if not(type(mo_f) is int or type(mo_f) is float) and not(type(mo_f) is np.ndarray):
        mo_f = np.array(mo_f)
        ko_f = np.array(ko_f)
        mp_f = np.array(mp_f)
        kp_f = np.array(kp_f)

    ma_aux = compute_6_by_6_diagonal_matrix_two_value(mp_f, mo_f)
    ka_aux = compute_6_by_6_diagonal_matrix_two_value(kp_f, ko_f)

    dp, do = compute_critical_damping(mo_f, ko_f, mp_f, kp_f)

    da_aux = compute_6_by_6_diagonal_matrix_two_value(dp, do)

    return ma_aux, ka_aux, da_aux


def rx(theta):
    """Rx's transformation.

        Parameters:
        theta (float): angle in radians.

        Returns:
        np.ndarray: The computed matrix

       """
    return np.matrix([[1, 0, 0],
                      [0, math.cos(theta), -math.sin(theta)],
                      [0, math.sin(theta), math.cos(theta)]])


def ry(theta):
    """Ry's transformation.

        Parameters:
        theta (float): angle in radians.

        Returns:
        np.ndarray: The computed matrix

       """
    return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                      [0, 1, 0],
                      [-math.sin(theta), 0, math.cos(theta)]])


def rz(theta):
    """Rz's transformation.

        Parameters:
        theta (float): angle in radians.

        Returns:
        np.ndarray: The computed matrix

       """
    return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0],
                      [0, 0, 1]])


def compute_ZYZ_trans(theta, theta2, theta3):
    """Compute the ZYZ transformation matrix based on the given angles.

        Parameters:
        theta (float): X-axis TCP position.
        theta2 (float): Y-axis TCP position.
        theta3 (float): Z-axis TCP position.

        Returns:
        np.ndarray: The computed transformation

       """

    m1 = rz(theta)
    m2 = ry(theta2)
    m3 = rz(theta3)

    mt = m1*m2*m3
    return mt


def compute_matrix_only_position(x, y, z):
    """Compute the transformation matrix based only on positions.
        The rotation will be [0, 0, 0]

        Parameters:
        x (float): X-axis TCP position.
        y (float): Y-axis TCP position.
        z (float): Z-axis TCP position.

        Returns:
        np.ndarray: The computed transformation matrix

       """
    mat = np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return mat


def expand_matrix(mat, req_row_size, req_colum_size):
    """Expand the size of a given matrix with dimensions a x b to req_row_size x req_colum_size.

        Parameters:
        mat (np.ndarray): Matrix to be transformed.
        req_row_size (int): Required row size
        req_colum_size (int): Required colum size

        Returns:
        np.ndarray: The computed matrix

        """
    if not (type(mat) is np.ndarray):
        mat = np.array(mat)

    aux = mat
    zero_arr1 = None
    zero_arr2 = None

    if len(mat) < req_row_size:
        req_size1 = req_row_size - len(mat)
        zero_arr2 = np.array([[0] * req_colum_size] * req_size1)

    if len(mat[0]) < req_colum_size:
        req_size2 = req_colum_size - len(mat[0])
        zero_arr1 = np.array([[0] * req_size2] * len(mat))

    if zero_arr1 is not None:
        aux = np.concatenate((aux, zero_arr1), axis=1)

    if zero_arr2 is not None:
        aux = np.concatenate((aux, zero_arr2))

    return aux
