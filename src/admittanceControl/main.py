
import math
from admittanceControl import *
from utilities import comp_trans, compute_parameters_matrix

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


def main():
    trans_i, comb_trans_i = comp_trans(q_i, DH_Parameters_a, DH_Parameters_d, DH_Parameters_alpha)
    trans_d, comb_trans_d = comp_trans(q_d, DH_Parameters_a, DH_Parameters_d, DH_Parameters_alpha)

    ma_aux, ka_aux, da_aux = compute_parameters_matrix(mo, ko, mp, kp)

    admittance_control = AdmittanceControl(ma_aux, ka_aux, da_aux, comb_trans_d[2], comb_trans_i[2], "euler")

    for i in np.arange(0.0, tmax, dt):

        if 5 <= i < 10:
            tau = [[1], [2], [3], [0], [0], [0]]
        elif 15 <= i < 20:
            tau = [[0], [0], [0], [1], [0.5], [1]]
        else:
            tau = [[0]] * 6

        admittance_control.step(dt, np.array(tau), True)


if __name__ == '__main__':
    main()
