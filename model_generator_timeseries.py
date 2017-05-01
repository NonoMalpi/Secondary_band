import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from itertools import compress

from sklearn.metrics import mean_squared_error, mean_absolute_error


class generate_df(object):
	
	def __init__(self, path_dict, main_name, date):
		self.__raw_df_list = { name : self.__import_files(path_df) for name, path_df in path_dict.items()}
		self.__df_dict = { name : self.__get_clean_data(df, name, date) for name, df in self.__raw_df_list.items()}
		self.name_list = list(path_dict.keys())
		self.main_name = main_name
		self.completed_df = self.__return_merged_df()

	def __import_files(self, path):
		df = pd.read_csv(path, encoding='latin1', delimiter=';')
		return df
	
	def __get_clean_data(self, df_original, name, date):
		weekday_dict = {
			0:'Wd', 1:'Wd', 2:'Wd', 3:'Wd', 4:'Wd', 5:'F', 6:'F' 
		}

		df = df_original.copy(deep=True)
		if name == 'spot':
			df = df[df['geoid'] == 3]
		df['date'] =  pd.to_datetime(df['datetime'].apply(lambda x: x[:10]), format='%Y-%m-%d')
		df['year'] = df['date'].dt.year
		df['time'] = pd.to_datetime(df['datetime'].apply(lambda x: x[:19])).dt.time
		df['month'] = df['date'].dt.month
		df['day'] = df['date'].dt.day
		df['hour'] = df['datetime'].apply(lambda x: x[11:13]).astype(int)
		df['minute'] = df['datetime'].apply(lambda x: x[14:16]).astype(int)
		df['weekday'] = df['date'].dt.dayofweek
		df.replace({'weekday':weekday_dict}, inplace=True)
		df['season'] = np.where(df['month'].isin(list(range(4,10))), 'summer', 'winter')
		df['date_hour'] = df.apply(lambda x: datetime.datetime.combine(x['date'], x['time']), axis=1)
		df.set_index('date_hour', inplace=True)
		df = df[df.index < date]
		clean_df = df[['date', 'year', 'month', 'season', 'day','weekday','time', 'hour', 'minute', 'value']]
		clean_df = clean_df[~clean_df.index.duplicated()]
		clean_df['hour'] = np.where(clean_df['hour'].isin(np.arange(9,23)), 'Peak', 'off_peak')
		clean_df = clean_df.rename(columns={'value':name})

		return clean_df

	def __return_merged_df(self):
		"""
		Return a merged df with all features, base_df refers to the name of the df
		to use as base of the merged df
		"""
		return self.__df_dict[self.main_name].join([self.__df_dict[name][[name]] for name in self.name_list if name != self.main_name])

	def return_raw_df_dict(self):
		return self.__raw_df_list

	def return_df_dict(self):
		return self.__df_dict

	def return_completed_df(self):
		"""
		Return the merged df with all raw features
		"""
		return self.completed_df

	def return_df_feature_engineering(self, name_list_24):
		""""
		Return the completed_df with feature engineering:
		Values from the past hour
		Values for the same hour from past day
		Moving averages
		"""
		df = self.completed_df[self.name_list + ['hour', 'weekday','season']]
		
		for feature in ['hour', 'weekday', 'season']:
			df[pd.get_dummies(df[feature], drop_first=True).columns.tolist()] = pd.get_dummies(df[feature], drop_first=True)

		custom_list_1_24 = [feature for feature in self.name_list if feature not in list(set([self.main_name] + name_list_24))]
		
		for feature in custom_list_1_24:
			df[feature + '-1'] = df[feature].shift(periods=1)
			df[feature + '-24'] = df[feature].shift(periods=24)
			df[feature + '_mov_aver_24'] = pd.rolling_mean(df[feature], window=24)

		for feature in name_list_24:
			df[feature + '-24'] = df[feature].shift(periods=24)

		df.dropna(subset=[self.main_name], inplace=True)

		return df


class timeseries_model(object):

	def __init__(self, df, features_list, output, start_date, end_date, train_length, pipeline, first_date_to_predict='2017-01-01',rolling=False):
		self.df = self.__get_df(df, start_date)
		self.features_list = features_list
		self.output = output
		self.date_index = self.__get_date_index(start_date, end_date)
		self.length = train_length
		self.pipeline = pipeline
		self.date_to_predict = first_date_to_predict
		self.rolling_bool = rolling
		self.cv = self.__create_cv_indexes()
		self.X, self.Y = self.__divide_features_output()

	def __get_df(self, df, start_date):
		df = df[df.index >= start_date]
		return df

	def __get_date_index(self, start_date, end_date):
		date_index = pd.date_range(start=start_date, end=end_date, freq='D')
		return date_index

	def __create_cv_indexes(self):
		cv_list = list()
		first_date = np.flatnonzero(self.date_index == self.date_to_predict)[0] - self.length
		if self.rolling_bool:
			for i in range(first_date, len(self.date_index) - self.length):
				train_period = self.date_index.date[i:self.length+i]
				test_period = self.date_index.date[self.length+i]
				train_index = self.df.reset_index()[self.df.reset_index()['date_hour'].dt.date.isin(train_period)].index
				test_index = self.df.reset_index()[self.df.reset_index()['date_hour'].dt.date == (test_period)].index
				train_test_list = [train_index, test_index]
				cv_list.append(train_test_list)
		else:
			for i in range(first_date, len(self.date_index) - self.length):
				train_period = self.date_index.date[:self.length+i]
				test_period = self.date_index.date[self.length+i]
				train_index = self.df.reset_index()[self.df.reset_index()['date_hour'].dt.date.isin(train_period)].index
				test_index = self.df.reset_index()[self.df.reset_index()['date_hour'].dt.date == (test_period)].index
				train_test_list = [train_index, test_index]
				cv_list.append(train_test_list)
		
		return cv_list

	def __divide_features_output(self):
		X = self.df[self.features_list].values
		Y = self.df[self.output].values
		return X, Y

	def obtain_cv_scores(self):
		CV_mae = list()
		CV_mse = list()
		output_list = list()

		for fold, (train_index, test_index) in enumerate(self.cv):
			if ((fold+1) % 10) == 0:
				print('Acting on fold %i' %(fold+1))
			self.pipeline.fit(self.X[train_index], self.Y[train_index])
			output_pred = self.pipeline.predict(self.X[test_index])
			CV_mae.append(mean_absolute_error(self.Y[test_index], output_pred))
			CV_mse.append(mean_squared_error(self.Y[test_index], output_pred))
			output_list.append(output_pred)

		output = np.concatenate(output_list)

		print('Mean absolute error: %0.4f +- %0.4f' %(np.mean(CV_mae), 2*np.std(CV_mae)))
		print('Mean squared error: %0.4f +- %0.4f' %(np.mean(CV_mse), 2*np.std(CV_mse)))

		self.CV_mae = CV_mae
		self.CV_mse = CV_mse

		first_date = np.flatnonzero(self.df.index >= self.date_to_predict)[0]

		self.result_df = pd.DataFrame({'y_true':self.Y[first_date:], 'y_pred':output},
								index=self.df.iloc[first_date:].index)
		self.result_df['error'] = self.result_df['y_true'] - self.result_df['y_pred']

	def plot_histogram_error(self):
		fig, ax = plt.subplots(1,1, figsize=(10,7))
		self.result_df.hist(column='error', bins=int(np.sqrt(len(self.result_df))), ax=ax,
							normed=True)

	def get_feature_importance(self):
		
		try:
			if hasattr(self.pipeline, 'steps'):
				if hasattr(self.pipeline.steps[-1][-1], 'feature_importances_'):
					fig, ax = plt.subplots(1,1, figsize=(10,20))
					pd.DataFrame(self.pipeline.steps[-1][-1].feature_importances_, index=self.features_list
						).sort(0, ascending=True).plot.barh(ax=ax, fontsize=16)
				elif hasattr(self.pipeline.steps[-1][-1], 'coef_'):
					if hasattr(self.pipeline.steps[-2][-1], 'get_support'):
						select_feat = self.pipeline.steps[-2][-1].get_support()
						df = pd.DataFrame([self.pipeline.steps[-1][-1].intercept_] + 
							self.pipeline.steps[-1][-1].coef_.tolist(), 
							index=['intercept'] + list(compress(self.features_list, select_feat)))
						print(df)
					else:
						df = pd.DataFrame([self.pipeline.steps[-1][-1].intercept_] + 
							self.pipeline.steps[-1][-1].coef_.tolist(), index=['intercept'] + 
							self.features_list)
						print(df)
			else:
				fig, ax = plt.subplots(1,1, figsize=(10,20))
				pd.DataFrame(self.pipeline.feature_importances_, index=self.features_list
					).sort(0, ascending=True).plot.barh(ax=ax, fontsize=16)
		except:
			raise ValueError('pipeline has not been fitted.')