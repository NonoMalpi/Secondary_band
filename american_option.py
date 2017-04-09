import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class american_option(object):
	def __init__(self, df, M, I, std_df):
		self.df = df
		self.M = M
		self.I = I
		self.std_df = std_df

	def __generate_sn(self, M,I,std):
		"""
		Function that generates random numbers, matching first moment, i.e. mean=0
		M: Length of the simulation process
		I: Number of paths to be simulated
		std: Standard deviations of the desired normal random number 
		"""
		sn = np.random.normal(loc=0, scale=std, size=(M, I/2))
		sn = np.concatenate((sn, -sn), axis=1)
		return sn

	def __compute_mcs_amer_option(self, S,K, pol_degree):
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

	def plot_real_predicted_values(self, date):
		#Plot real and predicted values from model without noise
		self.__index = np.flatnonzero(self.df.index == date)[0]
		self.df.iloc[self.__index:self.__index+self.M][['S', 'S_pred']].plot(figsize=(12,4))

	def generate_random_paths(self, date):
		#Compute the random paths for the desired date
		self.__index = np.flatnonzero(self.df.index == date)[0]
		base_mc = self.df.iloc[self.__index:self.__index+self.M]['S_pred'].values
		###Random paths
		#Obtain the std of the previous day
		self.__index_std = np.flatnonzero(self.std_df.index == date)[0] - 1
		std = self.std_df.iloc[self.__index_std]['std']
		estimations_mc = base_mc.reshape(-1,1) * np.exp(self.__generate_sn(self.M, self.I, std))
		estimations_mc = pd.DataFrame(data=estimations_mc, index=self.df.iloc[self.__index:self.__index+self.M].index)
		estimations_mc['mean'] = estimations_mc.mean(axis=1)
		estimations_mc['S_pred'] = base_mc
		estimations_mc['S_true'] = self.df.iloc[self.__index:self.__index+self.M]['S']
		self.df_mc = estimations_mc
		return self.df_mc

	def plot_montecarlo(self):
		#Plot a Monte Carlo sample of the paths generated
		fig, ax = plt.subplots(1,1, figsize=(20,8))
		self.df_mc.iloc[:, :int(self.I/100)].plot(alpha=0.2, legend=False, ax=ax)
		self.df_mc[['S_true', 'S_pred']].plot(lw=2.5, ax=ax, color=('red','blue'))

	def compute_option_value (self, K, pol_degree):
		#Compute the option value for a given strike (electric valuation) or a list of strikes
		S = self.df_mc.iloc[:,:self.I].values
		if isinstance(K, int) or isinstance(K, float):
			return self.__compute_mcs_amer_option(S,K,pol_degree)
		elif isinstance(K, np.ndarray):
			C_list = list()
			for k in K:
				C_list.append(self.__compute_mcs_amer_option(S,k,pol_degree))
			return C_list
		else:
			raise ValueError('K must be a number or an array of numbers')

	def plot_option_value_vs_strike(self, K_list, C_list):
		#Plot delta of the option 
		fig, axis = plt.subplots(1,1, figsize=(10,7))
		axis.plot(K_list, C_list)
		axis.set_title('Option value against Strike')
		axis.set_xlabel('Strike price $K$')
		axis.set_ylabel('Option value $€/MW$')