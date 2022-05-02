import math
import numpy as np

def rotToQuat(rot: np.array):
    q = np.zeros((4,1))
    q[0] = math.sqrt((1 + rot[0,0] + rot[1,1] + rot[2,2])/2)
    q[1] = (rot[2,1] - rot[1,2])/(4*q[0])
    q[2] = (rot[0,2] - rot[2,0])/(4*q[0])
    q[3] = (rot[1,0] - rot[0,1])/(4*q[0])

    return q

def normQuat(quat:np.array):
    return quat/math.sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)