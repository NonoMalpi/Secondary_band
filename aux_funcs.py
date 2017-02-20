import numpy as np
import pandas as pd

def generate_sn(M,I,std):
	"""
	Function that generates random numbers, matching first moment, i.e. mean=0
	M: Length of the simulation process
	I: Number of paths to be simulated
	std: Standard deviations of the desired normal random number 
	"""
	sn = np.random.normal(loc=0, scale=std, size=(M, I/2))
	sn = np.concatenate((sn, -sn), axis=1)

	return sn

def compute_mcs_amer_option(S,K, pol_degree):
	"""
	Function that computes the value of an American Option
	through the Least-Squared Monte Carlo simulation algorithm
	S: Monte Carlo paths of the secondary band price
	K: Strike (intrinsic value that industrial consumer considers)
	pol_degree: degree of the polynomial for the least-square regression
	Returns:
	C0 : Estimated present value of the American Option
	"""
	#Case base simulation payoff
	h = np.maximum(S-K, 0)
	M = S.shape[0]
	I = S.shape[1]

	#LSM algorithm
	V = np.copy(h)
	#Iterate from M-1 to 0
	for t in range(M-2, -1, -1):
		#Least-square regression to estimate V[t+1] based on S[t]
		reg = np.polyfit(S[t], V[t+1], pol_degree)
		#For the coefficients obatined, evaluated reg*S[t] to obtain an expected value
		# of the continuation value C
		C =	np.polyval(reg, S[t])
		#Compute for each time stampt t, the value of the American Option, as the
		#maximum between the expected continuation value and the payoff h at t
		V[t] = np.where(C>h[t], V[t+1], h[t])
	#MCS estimator
	C0 = 1/I * np.sum(V[0])

	return C0