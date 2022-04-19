import math
import numpy as np

# Limit for decimals. A number under this limit will be taken as 0
limit_dec = math.pow(10, -15)


def separate_rotation_translation(matrix):
    rotation, translation = [], []
    error = True

    if len(matrix) == 4:
        if len(matrix[0]) == 4 and len(matrix[1]) == 4 and len(matrix[2]) == 4 and len(matrix[3]) == 4:
            error = False

    if not error:
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


def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, math.cos(theta), -math.sin(theta)],
                      [0, math.sin(theta), math.cos(theta)]])


def Ry(theta):
    return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                      [0, 1, 0],
                      [-math.sin(theta), 0, math.cos(theta)]])


def Rz(theta):
    return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0],
                      [0, 0, 1]])


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


def comp_trans(joint_config, DH_Parameters_a, DH_Parameters_d, DH_Parameters_alpha):
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


def compute_critical_dumping(mo_f, ko_f, mp_f, kp_f):
    if type(mo_f) is int or type(mo_f) is float:
        aux_do = 2*math.sqrt(mo_f*(ko_f+1))
        aux_dp = 2*math.sqrt(mp_f*(kp_f+1))
    else:
        if not(type(mo_f) is np.ndarray):
            mo_f = np.array(mo_f)
            ko_f = np.array(ko_f)
            mp_f = np.array(mp_f)
            kp_f = np.array(kp_f)

        aux_do = 2 * np.sqrt(mo_f * (ko_f + np.eye(3)))
        aux_dp = 2 * np.sqrt(mp_f * (kp_f + np.eye(3)))

    return aux_dp, aux_do


def compute_parameters_matrix(mo_f, ko_f, mp_f, kp_f):

    if not(type(mo_f) is int or type(mo_f) is float) and not(type(mo_f) is np.ndarray):
        mo_f = np.array(mo_f)
        ko_f = np.array(ko_f)
        mp_f = np.array(mp_f)
        kp_f = np.array(kp_f)

    ma_aux = compute_6_by_6_diagonal_matrix_two_value(mp_f, mo_f)
    ka_aux = compute_6_by_6_diagonal_matrix_two_value(kp_f, ko_f)

    dp, do = compute_critical_dumping(mo_f, ko_f, mp_f, kp_f)

    da_aux = compute_6_by_6_diagonal_matrix_two_value(dp, do)

    return ma_aux, ka_aux, da_aux
