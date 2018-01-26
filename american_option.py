
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class american_option(object):
    """ Generate a American option object to obtain the valuation.
    In addition, this class computes the Monte Carlo random paths and allow to
    plot the variation of the valuation with the strike.

    Attributes
    ----------
    df: pandas.DataFrame
        DataFrame with real and predicted hourly values for all study period.

    M: int
        Number of hours the simulation last, typically: 24 hours, i.e. 1 day.

    I: int
        Number of random paths to compute.

    std_df: pandas.DataFrame
        DataFrame with hourly standard deviation of in-sample residuals 
        from the ML model.

    df_mc: pandas.DataFrame
        DataFrame containing all random paths from Monte Carlo simulation for 
        a given day.
    """
    def __init__(self, df, M, I, std_df):
        self.df = df
        self.M = M
        self.I = I
        self.std_df = std_df

    def __generate_sn(self, M,I,std):
        """
        Generate random numbers, matching first moment, i.e. mean=0

        Parameters
        ----------
        M: int
            Length of the simulation process in hours
        I: int
            Number of paths to be simulated
        std: array-like, shape =[M,]
            Hourly standard deviations of the in-sample residuals.

        Returns
        -------
        sn: array-like, shape = [M, I]
            Random numbers drawn from normal distribution
        """
        sn = np.random.normal(loc=0, scale=std, size=(I//2, M))
        sn = np.concatenate((sn, -sn), axis=0).T

        return sn

    def __compute_mcs_amer_option(self, S,K, pol_degree):
        """
        Function that computes the value of an American Option through 
        the Least-Squared Monte Carlo simulation algorithm.

        Parameters
        ----------
        S: array-like, shape=[M, I]
            Monte Carlo paths of the secondary band price.

        K: int
            Strike (intrinsic value that consumer considers).

        pol_degree: int
            degree of the polynomial for the least-square regression.
        
        Returns
        -------
        C0 : float 
            Estimated present value of the American Option.
        """

        # case base simulation payoff
        h = np.maximum(S-K, 0)
        M = S.shape[0]
        I = S.shape[1]

        ########################### LSM ALGORITHM ###########################
        V = np.copy(h)
        # iterate from M-1 to 0
        for t in range(M-2, -1, -1):

            # least-square regression to estimate V[t+1] based on S[t]
            reg = np.polyfit(S[t], V[t+1], pol_degree)

            # for the coefficients obatined, evaluated reg*S[t] 
            # to obtain an expected value of the continuation value C
            C = np.polyval(reg, S[t])

            # compute for each timestamp t, the value of the American Option, 
            # as the maximum between the expected continuation value 
            # and the payoff h at t
            V[t] = np.where(C>h[t], V[t+1], h[t])

        # MCS estimator
        C0 = 1/I * np.sum(V[0])
        #####################################################################

        return C0

    def plot_real_predicted_values(self, date):
        """
        Plot real and predicted values from model for a desired date.

        Parameters
        ----------
        date: str, format='%Y-%m-%d'
            Date to plot hourly secondary band prices.
        """

        self.__index = np.flatnonzero(self.df.index == date)[0]
        self.df.iloc[
            self.__index:self.__index+self.M
        ][['S', 'S_pred']].plot(figsize=(12,4), color=['blue', 'green'])

    def generate_random_paths(self, date):
        """
        Compute the random paths using Monte Carlo simulation for a desired date.

        Parameters
        ----------
        date: str, format='%Y-%m-%d'
            Date to compute the random paths.

        Returns
        -------
        df_mc: pandas.DataFrame
            DataFrame containing the all random paths for specific date.
        """
        # Obtain the deterministic part coming from the predicitons of ML model.
        self.__index = np.flatnonzero(self.df.index == date)[0]
        base_mc = self.df.iloc[
            self.__index:self.__index+self.M
        ]['S_pred'].values

        ######################### RANDOM PATHS ###############################
        # obtain the volatility from std dev of in-sample residuals 
        # of previous day. 
        self.__col_std = np.flatnonzero(self.std_df.columns == date)[0] - 1
        std = self.std_df.iloc[:, self.__col_std].values

        # compute the random paths as base_mc * np.exp (random numbers)
        # this array will be of shape (M, I)
        estimations_mc = base_mc.reshape(-1,1) * np.exp(
            self.__generate_sn(self.M, self.I, std)
        )
        estimations_mc = pd.DataFrame(
            data=estimations_mc, 
            index=self.df.iloc[self.__index:self.__index+self.M].index
        )
        ######################################################################

        # add mean of random paths, true value and predicted from ML model.
        estimations_mc['mean'] = estimations_mc.mean(axis=1)
        estimations_mc['S_pred'] = base_mc
        estimations_mc['S_true'] = self.df.iloc[
            self.__index:self.__index+self.M
        ]['S']
        self.df_mc = estimations_mc
        return self.df_mc

    def plot_montecarlo(self, save_bool=False):
        """
        Plot a Monte Carlo sample of the paths generated.

        Parameters
        ----------
        save_bool: bool, default: False
            Whether to save the plot as pdf format.
        """
        fig, ax = plt.subplots(1,1, figsize=(10,7))
        # plot just hundredth part of all random paths. 
        self.df_mc.iloc[:, :int(self.I/100)].plot(
            alpha=0.1, legend=False, ax=ax, color='grey'
        )
        self.df_mc[['S_true', 'S_pred']].rename(columns={
                'S_true':'$S_{t}$', 
                'S_pred':'$\widehat{S_{t}}$'
        }).plot(lw=2.5, ax=ax, color=('#50b847','#00133e'))
        ax.set_xlabel('Hour', size=14)
        ax.set_ylabel('$€/MW$', size=14)
        if save_bool == True:
            plt.savefig(str(self.df_mc.index[0])[:10] + '_montecarlo.pdf', 
                bbox_inches='tight'
            )

    def compute_option_value(self, K, pol_degree):
        """
        Compute the option value for a given strike (electric valuation) or list
        of strikes.

        Parameters
        ----------
        K: int, list
            Strike or electric valuation in €/MW.

        pol_degree: int
            Degree of the polynomial for the least-square regression of 
            american option.

        Returns
        -------
        C: int, list
            Valuation of the american option for the strike given.
        """
        # obtain all random paths and compute american option value.
        S = self.df_mc.iloc[:,:self.I].values
        if np.isscalar(K):
            return self.__compute_mcs_amer_option(S,K,pol_degree)
        elif isinstance(K, np.ndarray):
            C_list = list()
            for k in K:
                C_list.append(self.__compute_mcs_amer_option(S,k,pol_degree))
            return C_list
        else:
            raise ValueError('K must be a number or an array of numbers')

    def plot_option_value_vs_strike(self, K_list, C_list, date, save_bool=False):
        """
        Plot option value against strike of the option.

        Parameters
        ----------
        K_list: array-like
            Array of strike prices for the american option.

        C_list: array-like
            Array of valuation of the american option for the strikes given.

        date: str, format='%Y-%m-%d'
            Date to indicate plot.

        save_bool: bool, default: False.
            Whether to save the plot as pdf format.
        """
        fig, axis = plt.subplots(1,1, figsize=(10,7))
        axis.plot(K_list, C_list, color='grey', lw=2.0)
        #axis.set_title('Option value against Strike')
        axis.set_xlabel('Strike price $K$', size=14)
        axis.set_ylabel('Option value $€/MW$', size=14)
        axis.margins(0)
        axis.set_ylim([0, np.round(np.max(C_list))])
        if save_bool == True:
            plt.savefig(str(date) +'value_strike.pdf', bbox_inches='tight')