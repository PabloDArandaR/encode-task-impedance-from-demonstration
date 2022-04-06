import numpy as np
import math
import transformation

def precomputeT(a = [0, -0.425, -0.3922, 0, 0, 0], d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996], alpha = [math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0]):
    '''
    Precompute transformations given the DH parameters. Default values from UR5e.
    input
        - a <list>: list with all the a DH parameters
        - d <list>: list with all the d DH parameters
        - alpha <list>: list with all the alpha DH parameters
    output
        - Ta <numpy array 6x4x4>: numpy array that contains the translation transformation given by a
        - Td <numpy array 6x4x4>: numpy array that contains the translation transformation given by d
        - Talpha <numpy array 6x4x4>: numpy array that contains the rotation transformation given by alpha
    '''
    Ta = []
    Td = []
    Talpha = []
    for i in range(0,6):
        # New arrays to insert
        new_a = np.array([[1, 0, 0, a[i]],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        new_d = np.array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, d[i]],
            [0, 0, 0, 1]])
        new_alpha = np.array([[1, 0, 0, 0],
                                [0, math.cos(alpha[i]), -math.sin(alpha[i]), 0],
                                [0, -math.sin(alpha[i]), math.cos(alpha[i]), 0],
                                [0, 0, 0, 1]])

        # Append new matrixes to store them in memory
        Ta.append(new_a)
        Td.append(new_d)
        Talpha.append(new_alpha)

    return Ta, Td, Talpha


def TfromConfig(Ta: np.array,Td: np.array,Talpha: np.array, theta:np.array):
    '''
    Calculate transformation matrixes given a configuration theta and precomputed transformation.
    input
        - Ta <list>: list with all the a transformations
        - Td <list>: list with all the d transformations
        - Talpha <list>: list with all the alpha transformations
        - theta <numpy array 6>: configuration to which the transformation is calculated
    output
        - A <numpy array 6x4x4>: numpy array that contains the transformations frame to frame
        - T <numpy array 6x4x4>: numpy array that contains the transformation from frame to base frame
    '''
    T = np.zeros((6,4,4))
    A = np.zeros((6,4,4))
    for i in range(6):
        T[i,:,:] = np.eye(4)
        A[i,:,:] = np.eye(4)
    print(f"Shape of T: {T.shape}")
    print(f"Shape of A: {A.shape}")
    for i in range(0,6):
        c = math.cos(theta[i])
        s = math.sin(theta[i])
        Ttheta = [[c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
        if i == 0:
            A[0,:,:] = np.matmul(Td[i], np.matmul(Ttheta, np.matmul(Ta[i], Talpha[i])))
            T[0,:,:] = A[0,:,:]
        else:
            A[i,:,:] = np.matmul(Td[i], np.matmul(Ttheta, np.matmul(Ta[i], Talpha[i])))
            T[i,:,:] = np.matmul(T[i-1,:,:],A[i,:,:])
    
    return A,T

def TToTransQuat(T:np.array, normalize=False):
    '''
    Transform transformation to translation vector and quaternion.
    input
        - T <numpy array, 4,4>: Transformation to converted
        - normalize <bool>: default False. True to normalize the quaternion, else not normalized.
    output
        - trans <numpy array 3>: translation vector obtained from the transformation
        - quat <numpy array 3>: quaternion obtained from the transformation
    '''
    trans = np.transpose(T[0:3,3])
    quat = transformation.rotToQuat(T[0:3,0:3])
    if normalize:
        quat =transformation.normQuat(quat)
    return trans, quat

def Jacobian(T_set:np.array):
    '''
    Transform transformation to translation vector and quaternion.
    input
        - T <numpy array, 6x4x4>: set of transformations to calculate the jacobian
    output
        - Jp <numpy array 3>: Translational jacobian
        - Jo <numpy array 3>: Orientational jacobian
    '''
    Jp = np.zeros((6,3,6))
    Jo = np.zeros((6,3,6))
    for i in range(6):
        for j in range(i+1):
            if j == 0:
                Jp[i,0:3,j] = np.cross(np.array([0,0,1]),T_set[i,0:3,3])
                Jo[i,0:3,j] = np.array([0,0,1])
            else:
                Jp[i,0:3,j] = np.cross(T_set[j-1,0:3,2],(T_set[i,0:3,3] - T_set[j-1,0:3,3]))
                Jo[i,0:3,j] = T_set[j-1,0:3,2]
    
    return Jp, Jo

def velocity(Jp, Jo, dq):
    '''
    Obtain velocity and angular velocity of the frames given by Jp and Jo.
    input
        - Jp <numpy array, 3x6>: translational jacobian
        - Jo <numpy array, 3x6>: orientational jacobian
        - dq <numpy array, 6>: joint velocity
    output
        - v <numpy array 3>: velocity
        - w <numpy array 3>: angular velocity
    '''
    v = np.matmul(Jp, dq)
    w = np.matmul(Jo, dq)

    return v,w


if __name__ == "__main__":
    config = np.array([1,0,0,0,0,0])
    dconfig = np.array([2,0,math.pi,0,math.pi,0])

    
    print("-------------------------------------------------------------------------------- Precompute transformations -----------------------------------------------")
    Ta, Td, Talpha = precomputeT()
    
    print("-------------------------------------------------------------------------------- Transformation -----------------------------------------------------------")
    A,T = TfromConfig(Ta=Ta, Td=Td, Talpha=Talpha, theta=config)
    print(T)

    print("-------------------------------------------------------------------------------- Quaternion ---------------------------------------------------------------")
    _,quat = TToTransQuat(T[5,:,:])
    print(quat)
    print(f" norm is:  {math.sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)}")
    print(quat)
    print(f" norm is:  {math.sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)}")

    print("-------------------------------------------------------------------------------- Jacobian ---------------------------------------------------------------")
    Jp, Jo = Jacobian(T)
    print(f" Translation Jacobians: {Jp}")
    print(f" Orientation Jacobians: {Jo}")

    print("-------------------------------------------------------------------------------- End effector velocity --------------------------------------------------")
    v, w = velocity(Jp[5,:,:], Jo[5,:,:], dq = dconfig)
    print(f" Velocity: {v}")
    print(f" Angular velocity: {w}")
