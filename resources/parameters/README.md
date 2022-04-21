# Estimated Parameter 

This folder contains estimated parameter, desired trajectory, and corresponding time. For each task, there are two saparate files: "xdesired.npy" and "estparam.npy".

## xdesired.npy

This file contain a numpy array of size (N+1,T), where N denotes the number of degree of freedom (e.g., N = 3 for x, y, and z) and T denotes the number of the sampling points.

## estparam.npy

This file contain a numpy array of size (K,M,T,6), where K denotes the number of estimated parameter (in this order: Kp (stiffness matrix), Kv (dampping coefficient), and Im (mass/inertia matrix)), M denotes the number of trails/demonstrations, T denotes the number of the sampling points, and 6 denotes the number of the elements in the upper diagonal part of the 3x3 matrix. Note that, the data are now for x, y, and z.
