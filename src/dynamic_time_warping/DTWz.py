import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from scipy.signal import savgol_filter

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
			xs[n][m] = np.interp(t,np.linspace(0,1,xs[n][m].shape[0]),xs[n][m])+np.random.normal(0,0.0,t.shape[0])
	return xs

def preprocess_dtw_input(xs,Td=1000,length_equalize=True,start_equalize=True,amplitude_equalize=True):
	'''
	generate data for dtw using template of signals
	input:
		- xs <n x m, list of numpy array> : input signals where n denotes the number of signal and m denotes the number of repeatation
		- Td <int> : desired time interval of the output signals
		- length_equalize <bool> : perform signal length equalization or not
		- start_equalize <bool> : perform starting and ending points equalization or not
		- amplitude_equalize <bool> : perform amplitude equalization or not   
	output:
		- xs <n x m, list of numpy array> : preprocessed signals where n denotes the number of signal and m denotes the number of repeatation 
	'''

	# equalize signal length
	if length_equalize:
		for n in range(len(xs)):
			for m in range(len(xs[n])):
				t = np.linspace(0,1,Td)
				xs[n][m] = np.interp(t,np.linspace(0,1,xs[n][m].shape[0]),xs[n][m])
		xs = np.array(xs)

	# equalize starting and ending points
	if start_equalize:
		start_avg = np.zeros(len(xs))
		end_avg = np.zeros(len(xs))
		# compute start and end everage
		for n in range(len(xs)):
			for m in range(len(xs[n])):
				start_avg[n] += xs[n][m][0]/len(xs)
				end_avg[n] += xs[n][m][-1]/len(xs)
		# scale the signal
		for n in range(len(xs)):
			for m in range(len(xs[n])):
				#offset = np.linspace(xs[n][m][0]-start_avg[n],xs[n][m][-1]-end_avg[n],len(xs[n][0]))
				xs[n][m] = (xs[n][m]-xs[n][m][0])*(end_avg[n]-start_avg[n])/(xs[n][m][-1]-xs[n][m][0])+xs[n][m][0]

	# equalize signal amplitude
	if amplitude_equalize:
		sigmax = np.zeros(len(xs))
		sigmin = np.zeros(len(xs))
		for n in range(len(xs)):
			for m in range(len(xs[n])):
				sigmax += np.max(xs[n][m])/len(xs)
				sigmin += np.min(xs[n][m])/len(xs)
		for n in range(len(xs)):
			for m in range(len(xs[n])):      
				posgain = sigmax[n]/np.max(xs[n][m])
				xs[n][m][xs[n][m]>0] = posgain*xs[n][m][xs[n][m]>0]
				neggain = sigmin[n]/np.min(xs[n][m])
				xs[n][m][(xs[n][m]<0)] = np.abs(neggain)*xs[n][m][(xs[n][m]<0)]

	xs_ = np.zeros((len(xs),len(xs[0]),len(xs[0][0])))
	for n in range(len(xs)):
		for m in range(len(xs[n])):
			xs_[n][m][:] = xs[n][m]
	return xs_


def compute_euclidean_distance_matrix(x, y) -> np.array:
	"""Calculate distance matrix
	This method calcualtes the pairwise Euclidean distance between two sequences.
	The sequences can have different lengths.
	input:
		- x <numpy array of length l> : template signal
		- y <numpy array of length l> : input signal (that we want to perform DTW)
	output:
		- dist <numpy array of size l x l> : distance matrix
	"""
	dist = np.zeros((len(y), len(x)))
	for i in range(len(y)):
		for j in range(len(x)):
			dist[i,j] = np.sum((x[j,:]-y[i,:])**2)
	return dist

def compute_accumulated_cost_matrix(x, y, k= None) -> np.array:
	"""Compute accumulated cost matrix for warp path using Euclidean distance
	input:
		- x <numpy array of length l> : template signal
		- y <numpy array of length l> : input signal (that we want to perform DTW)
		- k <int> : local constrain
	output:
		- dist <numpy array of size l x l> : distance matrix
	"""
	distances = compute_euclidean_distance_matrix(x, y)
	'''cost = traceback_mat
	
	if k is not None:
		idx = np.indices((len(y), len(x)))
		mask = np.zeros((len(y), len(x)))
		baseline = np.argmin(cost,0)#+[np.argmin(cost,0)[-1]]#idx[0,:,:]
		baseline =np.array([baseline]).transpose()
		mask[np.logical_or((baseline-k//2)>idx[1,:,:],(baseline+k//2)<idx[1,:,:])] += np.inf
		traceback_mat = mask.transpose()+cost'''
	return distances

	# Initialization
	cost = np.zeros((len(y), len(x)))
	cost[0,0] = distances[0,0]
	
	for i in range(1, len(y)):
		cost[i, 0] = distances[i, 0] + cost[i-1, 0]  
		
	for j in range(1, len(x)):
		cost[0, j] = distances[0, j] + cost[0, j-1]  

	# Accumulated warp path cost
	for i in range(1, len(y)):
		for j in range(1, len(x)):
			cost[i, j] = min(
				cost[i-1, j],    # insertion
				cost[i, j-1],    # deletion
				cost[i-1, j-1]   # match
			) + distances[i, j] 
	if k is not None:
		idx = np.indices((len(y), len(x)))
		mask = np.zeros((len(y), len(x)))
		baseline = np.argmin(cost,0)#+[np.argmin(cost,0)[-1]]#idx[0,:,:]
		baseline =np.array([baseline]).transpose()
		mask[np.logical_or((baseline-k//2)>idx[1,:,:],(baseline+k//2)<idx[1,:,:])] += np.inf
		cost = mask.transpose()+cost
	return cost



def dp(dist_mat, k= None):
	"""
	Find minimum-cost path through matrix `dist_mat` using dynamic programming.

	The cost of a path is defined as the sum of the matrix entries on that
	path. See the following for details of the algorithm:

	- http://en.wikipedia.org/wiki/Dynamic_time_warping
	- https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

	The notation in the first reference was followed, while Dan Ellis's code
	(second reference) was used to check for correctness. Returns a list of
	path indices and the cost matrix.
	"""

	N, M = dist_mat.shape
	
	# Initialize the cost matrix
	cost_mat = np.zeros((N + 1, M + 1))
	for i in range(1, N + 1):
		cost_mat[i, 0] = np.inf
	for i in range(1, M + 1):
		cost_mat[0, i] = np.inf

	# Fill the cost matrix while keeping traceback information
	traceback_mat = np.zeros((N, M))
	for i in range(N):
		for j in range(M):
			penalty = [
				cost_mat[i, j],      # match (0)
				cost_mat[i, j + 1],  # insertion (1)
				cost_mat[i + 1, j]]  # deletion (2)
			i_penalty = np.argmin(penalty)
			cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
			traceback_mat[i, j] = i_penalty

	'''cost = traceback_mat

	if k is not None:
		idx = np.indices((N, M))
		mask = np.zeros((N, M))
		baseline = np.argmin(cost,0)#+[np.argmin(cost,0)[-1]]#idx[0,:,:]
		baseline =np.array([baseline]).transpose()
		mask[np.logical_or((baseline-k//2)>idx[1,:,:],(baseline+k//2)<idx[1,:,:])] += np.inf
		traceback_mat = mask.transpose()+cost'''
	
	# Traceback from bottom right
	i = N - 1
	j = M - 1
	path = [(i, j)]
	while i > 0 or j > 0:
		tb_type = traceback_mat[i, j]
		if tb_type == 0:
			# Match
			i = i - 1
			j = j - 1
		elif tb_type == 1:
			# Insertion
			i = i - 1
		elif tb_type == 2:
			# Deletion
			j = j - 1
		path.append((i, j))

	# Strip infinity edges from cost_mat before returning
	cost_mat = cost_mat[1:, 1:]
	return (path[::-1], cost_mat)

def perform_dtw(xs,k=None,filter=(11,3)):
	"""Compute accumulated cost matrix for warp path using Euclidean distance on numpy array of all signals
	input:
		- xs <n x m, list of numpy array> : input signals where n denotes the number of signal and m denotes the number of repeatation
		- k <int> : local constrain
	output:
		- xs <n x m, list of numpy array> : output signals where n denotes the number of signal and m denotes the number of repeatation
	"""
	xs_ = deepcopy(xs)
	
	temp = []
	for n in range(len(xs)):
		#temp.append(xs[n][0])
		temp.append(np.median(xs[n],0))
	temp = np.array(temp)
	temp = temp.transpose()

	others = []
	for n in range(len(xs)):
		others_i = []
		for m in range(0,len(xs[n])):
			others_i.append(xs[n][m])
		others.append(others_i)
	others = np.array(others)
	others = np.swapaxes(others,0,1)
	others = np.swapaxes(others,1,2)


	R = []
	for m in range(0,others.shape[0]): 	
		r = compute_accumulated_cost_matrix(temp,others[m,:,:],k=k)
		path, cost_mat = dp(r,k=k)
		y_path = [p[1] for p in path]


		for n in range(len(xs)):
			t = np.linspace(0,1,xs_[n][m].shape[0])
			xs_[n][m] = deepcopy(np.interp(t,np.linspace(0,1,len(y_path)),xs[n,m,y_path]))
			if filter is not None:
				xs_[n][m] = savgol_filter(xs_[n][m], filter[0], filter[1])
			#xs_[n][m] = deepcopy(xs[n,m,:100])
		R.append(r)

		if 0:
			plt.title("Cost matrix")
			plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
			x_path, y_path = zip(*path)
			plt.plot(y_path, x_path)
			plt.show()
	R = np.array(R)
	print(R.shape,others.shape[0])
	return xs_, R
