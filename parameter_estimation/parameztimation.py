import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy, copy
import torch
import sys
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D
import dynesty, nestle
import random

def generate_dtw_data(xs,To=100):
	'''
	generate data for dtw using template of signals
	input:
		- xs <n x m, list of numpy array> : input signals where n denotes the number of signal and m denotes the number of repeatation
		- To <int> : time scale multiplication, the output signals are with the size of To*m
	output:
		- xs <n x m, list of numpy array> : generate signals where n denotes the number of signal and m denotes the number of repeatation 
	'''
	for n in range(len(xs)):
		for m in range(len(xs[n])):
			xs[n][m] = np.array(xs[n][m])
			t = np.linspace(0,1,To*xs[n][m].shape[0])
			xs[n][m] = np.interp(t,np.linspace(0,1,xs[n][m].shape[0]),xs[n][m])+np.random.normal(0,0.05,t.shape[0])
	return xs

def spd_check(matrix):
	'''
	check whether the input matrix is a semi-positive definite matrix or not
	input:
		- matrix <n x n, numpy array> : input matrix (square)
	output:
		- spd <bool> :  true  = the input matrix is a semi-positive definite matrix
						false = the input matrix is not a semi-positive definite matrix
	'''
	spd = True
	if np.sum(np.abs(matrix-matrix.transpose())) != 0:
		spd = False
		print("non-symetric")
	lamb, e = np.linalg.eig(matrix)
	if (np.sum(np.iscomplex(lamb)) > 0) or (np.sum(lamb < 0) > 0):
		spd = False
		print("complex or negative eigenvalue")
	return spd

def spd_correction(inputmatrix):
    '''
    perform semi-positive definite correction, find a nearest-SPD matrix
    '''
    ipmatrix = inputmatrix.clone().detach()
    B = 0.5*(ipmatrix + torch.transpose(ipmatrix, 1, 0))
    U, S, V = torch.svd(B)
    S = torch.diag(S)
    H = torch.matmul(torch.matmul(V,S),torch.transpose(V,1,0))
    ipmatrix = 0.5*(B+H).detach()
    ipmatrix.requires_grad = True
    return ipmatrix

def positive_correction(inputmatrix):
    '''
    convert all negative element to zero 
    '''
    ipmatrix = inputmatrix.clone().detach()
    ipmatrix[ipmatrix < 0] = torch.abs(ipmatrix[ipmatrix < 0]*0)
    ipmatrix.requires_grad = True
    return ipmatrix

def estimate_parameter(tim,pos_avg,pos,vel,accel,force,window_size=20,estimatedKp=None,estimatedKv=None,estimatedIm=None,loss_threshold=0.005,learning_rate=1,correction=True,verbose=False):
    '''
    estimate stiffness coefficient matrix (Kp), damping coefficient matrix (Kv), and inertial/mass matrix (Im) using the data including average position trajectory, position trajectory,
    velocity profile, acceleration profile, and force profile
    input:
        - tim <t , numpy array> : time array where t denotes the number of the sampling points
        - pos_avg <n x t, numpy array> : average position array of size n x t, where n denotes the number of degree of freedom and t denotes the sampling points
        - pos <n x m x t, numpy array> : position trajectory of size n x m x t, where n denotes the number of degree of freedom, m denotes the number of trails,
                                         and t denotes the sampling points
        - vel <n x m x t, numpy array> : velocity profile of size n x m x t, where n denotes the number of degree of freedom, m denotes the number of trails,
                                         and t denotes the sampling points
        - accel <n x m x t, numpy array> : acceleration profile of size n x m x t, where n denotes the number of degree of freedom, m denotes the number of trails,
                                         and t denotes the sampling points
        - force <n x m x t, numpy array> : force profile of size n x m x t, where n denotes the number of degree of freedom, m denotes the number of trails,
                                         and t denotes the sampling points
        - window_size <int> : window size corresponding to each estimated Kp, Kv, and Im, this should be 0 < window_size < t
        - estimatedKp <None or numpy array of size n x n> : set to None if you want to estimate Kp, otherwise specify fixed Kp, where n denotes the number of degree of freedom
        - estimatedKv <None or numpy array of size n x n> : set to None if you want to estimate Kv, otherwise specify fixed Kv, where n denotes the number of degree of freedom
        - estimatedIm <None or numpy array of size n x n> : set to None if you want to estimate Im, otherwise specify fixed Im, where n denotes the number of degree of freedom
        - loss_threshold <float> : for each window, the learning stops when the loss change is below the threshold
        - learning_rate <float> : learning rate
        - correction <bool> : set to True if you want to perform spd and positive correction after each learning iteration
        - verbose <bool> : set to True if you want to print fitting statistic after estimate the parameter using each window
    output:
        - Kps <w x n x n, numpy array> : estimated stiffness coefficient matrix of size w x n x n, where w denotes the number of window applied and n denotes the number of 
                                         degree of freedom
        - Kvs <w x n x n, numpy array> : estimated damping coefficient matrix of size w x n x n, where w denotes the number of window applied and n denotes the number of 
                                         degree of freedom
        - Ims <w x n x n, numpy array> : estimated inertial/mass matrix of size w x n x n, where w denotes the number of window applied and n denotes the number of 
                                         degree of freedom
    '''
    
    # output list (converted to numpy in the final step)
    ndim = pos.shape[0]
    Kps = np.zeros((tim.shape[0],ndim,ndim))
    Kvs = np.zeros((tim.shape[0],ndim,ndim))
    Ims = np.zeros((tim.shape[0],ndim,ndim))
    
    pbar = tqdm(total=len(tim)-window_size) # initialize a progress bar
    for ti in range(0,len(tim)-window_size): # loop to each window size
        
        # crop the signals using the window size 
        e = np.expand_dims(pos_avg[:,ti:ti+window_size],1)-pos[:,:,ti:ti+window_size]
        v = vel[:,:,ti:ti+window_size]
        a = accel[:,:,ti:ti+window_size]
        f = force[:,:,ti:ti+window_size]
        
        # convert to torch cuda tensor
        e = torch.FloatTensor(e.reshape(3,e.shape[1]*e.shape[2])).cuda()
        v = torch.FloatTensor(v.reshape(3,v.shape[1]*v.shape[2])).cuda()
        a = torch.FloatTensor(a.reshape(3,a.shape[1]*a.shape[2])).cuda()
        f = torch.FloatTensor(f.reshape(3,f.shape[1]*f.shape[2])).cuda()
        
        
        # define the training parameter
        Kp_ = torch.autograd.Variable(torch.zeros(ndim, ndim).type(torch.float32), requires_grad=True).cuda() if estimatedKp is None else torch.FloatTensor(estimatedKp).cuda() 
        Kv_ = torch.autograd.Variable(torch.zeros(ndim, ndim).type(torch.float32), requires_grad=True).cuda() if estimatedKv is None else torch.FloatTensor(estimatedKv).cuda() 
        Im_ = torch.autograd.Variable(torch.zeros(ndim, ndim).type(torch.float32), requires_grad=True).cuda() if estimatedIm is None else torch.FloatTensor(estimatedIm).cuda() 
        
        # use the parameter from the previous learning iteration as the initial guess
        if ti != 0:
            Kp_ = Kp_*0+torch.FloatTensor(Kps[ti-1,:,:]).cuda() if estimatedKp is None else Kp_
            Kv_ = Kv_*0+torch.FloatTensor(Kvs[ti-1,:,:]).cuda() if estimatedKv is None else Kv_
            Im_ = Im_*0+torch.FloatTensor(Ims[ti-1,:,:]).cuda() if estimatedIm is None else Im_
        
        # set the parameter to be in the gradient computation
        parameters = []
        if estimatedKp is None:
            parameters.append(Kp_)
        if estimatedKv is None:
            parameters.append(Kv_)
        if estimatedIm is None:
            parameters.append(Im_)
        
        previousloss = 1e10
        for i in range(10000): # learning for 10,000 iterations (in maximum)
            
            # try to fit -> Im*a-Kp*e+Kv*v = f
            y_ = f
            x_ = torch.matmul(Im_,a)-torch.matmul(Kp_,e)+torch.matmul(Kv_,v)
            
            # compute the mean-square error
            criterion = torch.nn.MSELoss()
            loss = criterion(input=x_, target=y_)
            
            # compute the gradient
            grd = torch.autograd.grad(loss,parameters)
            
            # update the parameters
            for pi in range(len(parameters)):
                parameters[pi] = parameters[pi] - learning_rate*grd[pi]
                
            # perform semi-positive-definite correction
            if correction:
                for pi in range(len(parameters)):
                    parameters[pi] = spd_correction(parameters[pi])                    
            
            # finalize the parameter update
            pi = 0
            if estimatedKp is None:
                Kp_ = parameters[pi]
                pi += 1
            if estimatedKv is None:
                Kv_ = parameters[pi]
                pi += 1
            if estimatedIm is None:
                # perform positive correction
                if correction:
                    parameters[pi] = positive_correction(parameters[pi])
                Im_ = parameters[pi]
                pi += 1
            
            # update the process bar
            pbar.set_description("loss: "+str(loss.item()))
            
            # stop learning if the loss decrease below the threshold
            if ((previousloss-loss.item()) < loss_threshold) and (previousloss > loss.item()):
                break
            previousloss = loss.item()
        
        # print fitting statistics
        if verbose:
            print("-----------------    summary    --------------------")
            print("final loss:",loss.item())
            print("\nestimated Kp")
            print(Kp_.cpu().detach().numpy())
            print("\nestimated Im")
            print(Im_.cpu().detach().numpy())
            print("\nestimated Kv")
            print(Kv_.cpu().detach().numpy())
            print("----------------------------------------------------")
        
        # store the estimated parameters
        Kps[ti,:,:] = Kp_.cpu().detach().numpy()
        Kvs[ti,:,:] = Kv_.cpu().detach().numpy()
        Ims[ti,:,:] = Im_.cpu().detach().numpy()
        
        # update the process bar
        pbar.set_description("loss: "+str(loss.item()))
        pbar.update(1)
    
    pbar.close()
    return Kps, Kvs, Ims

def reconstruct_trajectories(tim,avg_pos,force,Kps,Kvs,Ims, window_size = 10):
    '''
    reconstruct the trajectories using time, average position trajectory, force profile, and the estimated parameters
    input:
        - tim <t , numpy array> : time array where t denotes the number of the sampling points
        - pos_avg <n x t, numpy array> : average position array of size n x t, where n denotes the number of degree of freedom and t denotes the sampling points
        - force <n x m x t, numpy array> : force profile of size n x m x t, where n denotes the number of degree of freedom, m denotes the number of trails,
                                         and t denotes the sampling points
        - Kps <w x n x n, numpy array> : estimated stiffness coefficient matrix of size w x n x n, where w denotes the number of window applied and n denotes the number of 
                                         degree of freedom
        - Kvs <w x n x n, numpy array> : estimated damping coefficient matrix of size w x n x n, where w denotes the number of window applied and n denotes the number of 
                                         degree of freedom
        - Ims <w x n x n, numpy array> : estimated inertial/mass matrix of size w x n x n, where w denotes the number of window applied and n denotes the number of 
                                         degree of freedom
        - window_size <int> : window size corresponding to each estimated Kp, Kv, and Im, this should be 0 < window_size < t
        
    output:
        - exs <n x m x t, numpy array> : estimated position trajectory of size n x m x t, where n denotes the number of degree of freedom, m denotes the number of trails,
                                         and t denotes the sampling points
        - dexs <n x m x t, numpy array> : estimated velocity profile of size n x m x t, where n denotes the number of degree of freedom, m denotes the number of trails,
                                         and t denotes the sampling points
        - ddexs <n x m x t, numpy array> : estimated acceleration profile of size n x m x t, where n denotes the number of degree of freedom, m denotes the number of trails,
                                         and t denotes the sampling points
    '''
    
    # initialize reconstructed trajectories to zero
    n = force.shape[0]
    m = force.shape[1]

    exs = np.zeros((n,m,len(tim)))
    dexs = np.zeros((n,m,len(tim)))
    ddexs = np.zeros((n,m,len(tim)))
    
    for idx in range(m): # loop through each trails
        # set all initial condition to zero
        xie_ = np.zeros((n,1))
        vie_ = np.zeros((n,1))
        aie_ = np.zeros((n,1))
        
        for ti in range(0,len(tim)): # loop through each timestep
            
            # get estimated parameters for each timestep
            if ti < len(tim)-window_size:
                Kpe = Kps[ti,:,:]
                Ime = Ims[ti,:,:]
                Kve = Kvs[ti,:,:]
            else:
                Kpe = Kps[len(tim)-window_size-1,:,:]
                Ime = Ims[len(tim)-window_size-1,:,:]
                Kve = Kvs[len(tim)-window_size-1,:,:]
                
            # compute difference from the average trajectory at current timestep
            eie_ = avg_pos[:,ti].reshape((n,1))-xie_
            
            # compute time difference
            dt = tim[ti+1]-tim[ti] if ti != len(tim)-1 else tim[ti]-tim[ti-1]\
            
            # compute acceleration using force and the estimated parameters
            aie_ = np.matmul(np.linalg.inv(Ime), np.matmul(Kpe,eie_)-np.matmul(Kve,vie_)+np.expand_dims(force[:,idx,ti],-1))
            
            # integrate acceleration
            vie_ = vie_ + aie_*dt
            xie_ = xie_ + vie_*dt
            
            # store the estimated trajectories
            exs[:,idx,ti] = deepcopy(xie_).reshape((1,n))
            dexs[:,idx,ti] = deepcopy(vie_).reshape((1,n))
            ddexs[:,idx,ti] = deepcopy(aie_).reshape((1,n))
    return exs, dexs, ddexs
    

def plot_ellipsoid_3d(center,axes, ax):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""
    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))
    
    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = center + np.dot(axes, [x[i,j],y[i,j],z[i,j]])
    
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='#2980b9', alpha=0.03)
    
def plot_param_ellipsoid(ax,param,average_pos, overlay,step=1,tmax=100,window_size=10,gain=1):
    '''
    plot the parameter ellipsoid on the average position trajectory
    input
        - param <w x n x n> : parameter to be ploted where w denotes the number of window applied and n denotes the number of degree of freedom
        - pos_avg <n x t, numpy array> : average position array of size n x t, where n denotes the number of degree of freedom and t denotes the sampling points
        - overlay <n x m x t, numpy array> : overlaid position array of size n x m x t, where n denotes the number of degree of freedom,
                                             m denote the trial number, and t denotes the sampling points
        - step <int> : number of step
        - tmax <float> : maximum timestep
        - window_size <int> : window size
        - gain <float> : average position scaling gain
    '''
    
    # initialize array for storing sampled average trajectories
    trajs = []
    for i in range(average_pos.shape[0]):
        trajs.append([])
      
    for i in tqdm(range(0,tmax,step)):
        
        # use the estimated parameter as the covariance matrix
        if i < tmax-window_size:
            covm = param[i,:,:]
        else:
            covm = param[tmax-window_size-1,:,:]
            
        x = average_pos[:,i]
        
        # correct negative eigenvalue
        lamb, e = np.linalg.eig(covm)
        lamb[lamb<0] = 0
        covm = np.matmul(np.matmul(e,np.diag(lamb)),e.transpose())
        
        # perform eigenvalue decomposition and find the ellipsoid parameter
        lamb, evec = np.linalg.eig(covm)
        
        for j in range(average_pos.shape[0]):
            trajs[j].append(x[j])
        
        # plot the ellipsoid
        plot_ellipsoid_3d(np.array(x)*gain,evec*np.sqrt(lamb), ax)
    
    # plot the average trajectory
    trajs = np.array(trajs)*gain
    ax.plot(trajs[0],trajs[1],trajs[2],color="tab:blue")
    ax.set_xlabel("x",fontsize=18)
    ax.set_ylabel("y",fontsize=18)
    ax.set_zlabel("z",fontsize=18)
    
    # overlay the estimated trajectories
    for i in range(overlay.shape[1]):
        ax.plot(overlay[0,i]*gain,overlay[1,i]*gain,overlay[2,i]*gain,color="tab:blue",alpha=0.25)
    
    
    



