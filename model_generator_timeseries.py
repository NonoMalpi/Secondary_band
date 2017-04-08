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
		#if date == '2016':
		#	df = df[df.index < '2017']
		#elif date == '2017':
		#	df = df[df.index >= '2016-12-31']
		#else:
		#	pass
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

	def __init__(self, df, features_list, output, start_date, end_date, train_length, pipeline, rolling=False):
		self.df = self.__get_df(df, start_date)
		self.features_list = features_list
		self.output = output
		self.date_index = self.__get_date_index(start_date, end_date)
		self.length = train_length
		self.pipeline = pipeline
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
		
		if self.rolling_bool:
			for i in range(len(self.date_index) - self.length):
				train_period = self.date_index.date[i:self.length+i]
				test_period = self.date_index.date[self.length+i]
				train_index = self.df.reset_index()[self.df.reset_index()['date_hour'].dt.date.isin(train_period)].index
				test_index = self.df.reset_index()[self.df.reset_index()['date_hour'].dt.date == (test_period)].index
				train_test_list = [train_index, test_index]
				cv_list.append(train_test_list)
		else:
			for i in range(len(self.date_index) - self.length):
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


		self.result_df = pd.DataFrame({'y_true':self.Y[self.length*24:], 'y_pred':output},
								index=self.df.iloc[self.length*24:].index)
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



#ADD A DESIRED DATE TO PERFORM PREDICT JUST ON THIS DATE
class forecast_2017_samples(object):

	def __init__(self, df, feature_list, output, fitted_model, ar_param, ar_order, ma_param, ma_order, std):
		self.df = df
		self.feature_list = feature_list
		self.output = output
		self.model = fitted_model
		self.ar = ar_param
		self.ma = ma_param
		self.ar_order = ar_order
		self.ma_order = ma_order
		self.std = std
		self.X, self.Y = self.__divide_features_output()

	def __fill_arima(self, df):
		for time in range(len(self.ar)):
			df['zt-'+str(self.ar_order[time]) +'_ar'] = df['zt'].shift(self.ar_order[time]) * self.ar[time]

		for time in range(len(self.ma)):
			df['zt-'+str(self.ma_order[time]) + '_ma'] = df['zt'].shift(self.ma_order[time]) * self.ma[time]

		return df

	def __divide_features_output(self):
		X = self.df[self.feature_list].values
		Y = self.df[self.output].values
		return X, Y

	def get_2017_predictions_from_base_model(self):
		self.y_pred = self.model.predict(self.X)
		print('MAE: %.4f, MSE: %.4f' %(mean_absolute_error(self.Y, self.y_pred), mean_squared_error(self.Y, self.y_pred)))
		df = pd.DataFrame({'y_true': self.Y, 'y_pred': self.y_pred}, index=self.df.index)
		df['error'] = df['y_true'] - df['y_pred']
		self.z_df = df

		return self.z_df

	def get_2017_predictions_arima_effect(self):
		zt = (np.log1p(self.Y) - np.log1p(self.y_pred))
		model_arima_df = pd.DataFrame({'f(x)':self.y_pred, 'zt': zt, 'S': self.Y}, index=self.df.index)
		model_arima_df = self.__fill_arima(model_arima_df)
		model_arima_df['zt_pred'] = model_arima_df.iloc[:,-(len(self.ar) + len(self.ma)):].sum(axis=1)
		model_arima_df['S_pred'] = model_arima_df['f(x)'] * np.exp(model_arima_df['zt_pred'])

		print('MAE: %.4f, MSE: %.4f' %(mean_absolute_error(model_arima_df['S'], model_arima_df['S_pred']), 
			mean_squared_error(model_arima_df['S'], model_arima_df['S_pred'])))

		self.model_arima_df_base = model_arima_df.copy(deep=True)

		return model_arima_df

	def get_2017_predictions_with_noise(self):
		noise = np.random.normal(loc=0, scale=self.std, size=(len(self.Y)))
		self.model_arima_df_base['noise'] = noise
		self.model_arima_df_base['S_pred_noise'] = self.model_arima_df_base['S_pred'] * \
									np.exp(self.model_arima_df_base['noise'])

		print('MAE: %.4f, MSE: %.4f' %(mean_absolute_error(self.model_arima_df_base['S'], self.model_arima_df_base['S_pred_noise']), 
			mean_squared_error(self.model_arima_df_base['S'], self.model_arima_df_base['S_pred_noise'])))

		return self.model_arima_df_base