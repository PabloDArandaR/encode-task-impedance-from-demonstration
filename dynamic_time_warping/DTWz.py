import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy, copy

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
		for n in range(len(xs)):
			for m in range(len(xs[n])):
				start_avg[n] += xs[n][m][0]/len(xs)
				end_avg[n] += xs[n][m][-1]/len(xs)
		for n in range(len(xs)):
			for m in range(len(xs[n])):
				offset = np.linspace(xs[n][m][0]-start_avg[n],xs[n][m][-1]-end_avg[n],len(xs[n][0]))
				xs[n][m] = xs[n][m]-offset

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
	return xs


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
			dist[i,j] = (x[j]-y[i])**2
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

def perform_dtw(xs,k=None):
	"""Compute accumulated cost matrix for warp path using Euclidean distance on numpy array of all signals
	input:
		- xs <n x m, list of numpy array> : input signals where n denotes the number of signal and m denotes the number of repeatation
		- k <int> : local constrain
	output:
		- xs <n x m, list of numpy array> : output signals where n denotes the number of signal and m denotes the number of repeatation
	"""
	xs_ = deepcopy(xs)
	R = []
	for n in range(len(xs)):
		temp = xs[n][0]
		R.append([])
		for m in range(1,len(xs[n])):
			r = compute_accumulated_cost_matrix(temp,xs[n][m],k=k)
			xs_[n][m] = xs[n,m,np.argmin(r,0)]
			R[-1].append(r)
	return xs_, R