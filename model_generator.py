import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import KFold, GroupKFold, train_test_split
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
		if date == '2016':
			df = df[df.index < '2017']
		elif date == '2017':
			df = df[df.index >= '2016-12-31']
		else:
			pass
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

class train_model(object):

	def __init__(self, df, output, features_list, n_folds, cv_type='normal'):
		self.df = df
		self.output = output
		self.features_list = features_list
		self.n_folds = n_folds
		self.X, self.Y = self.__divide_features_output()
		self.cv_type = cv_type
		self.x_train, self.x_test, self.y_train, self.y_test = self._get_train_test_cv_type()

	def __divide_features_output(self):
		X = self.df[self.features_list].values
		Y = self.df[self.output].values
		return X, Y

	def _get_train_test_cv_type(self):
		if self.cv_type == 'group':
			gkf = GroupKFold(n_splits=self.n_folds)

			#Date as a group
			groups_for_train_test_split = self.df.index.date

			train_idx, test_idx = list(gkf.split(self.X, self.Y, groups=groups_for_train_test_split))[0]

			x_train, y_train = self.X[train_idx], self.Y[train_idx]
			x_test, y_test = self.X[test_idx], self.Y[test_idx]

			groups_for_cv = self.df.iloc[train_idx].index.date

			self.cv = list(gkf.split(x_train, y_train, groups_for_cv))
		else:
			kf = KFold(n_splits=self.n_folds, random_state=0, shuffle=True)

			x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, train_size=0.8, test_size=0.2, random_state=0)

			self.cv = list(kf.split(x_train, y_train))

		return x_train, x_test, y_train, y_test

	def obtain_cv_score(self, pipeline):

		CV_mse = list()
		CV_mae = list()

		for fold, (train_index, test_index) in enumerate(self.cv):
			print('Acting on fold %i' %(fold+1))
			pipeline.fit(self.x_train[train_index], self.y_train[train_index])
			if hasattr(pipeline, 'predict'):
				output_pred = pipeline.predict(self.x_train[test_index])
			else:
				raise ValueError('pipieline does not contain "predict" method') 
			CV_mae.append(mean_absolute_error(self.y_train[test_index], output_pred))
			CV_mse.append(mean_squared_error(self.y_train[test_index], output_pred))

		print('Mean absolute error: %0.4f +- %0.4f' %(np.mean(CV_mae), 2*np.std(CV_mae)))
		print('Mean squared error: %0.4f +- %0.4f' %(np.mean(CV_mse), 2*np.std(CV_mse)))

	def obtain_train_test_error(self, pipeline):
		pipeline.fit(self.x_train, self.y_train)
		
		if hasattr(pipeline, 'predict'):
			print('Train MAE: ' +str(mean_absolute_error(self.y_train, pipeline.predict(self.x_train))) +
				', Train MSE: ' +str(mean_squared_error(self.y_train, pipeline.predict(self.x_train))))

			print('Test MAE: ' +str(mean_absolute_error(self.y_test, pipeline.predict(self.x_test))) +	
				', Test MSE: ' +str(mean_squared_error(self.y_test, pipeline.predict(self.x_test))))
		else:
			raise ValueError('pipieline does not contain "predict" method')

		errors_df = pd.DataFrame({'y_true':self.y_test, 'y_pred':pipeline.predict(self.x_test)})
		errors_df['error'] = errors_df['y_true'] - errors_df['y_pred']

		self.pipeline = pipeline
		self.errors_df = errors_df

	def plot_histogram_error(self):

		try:
			fig, ax = plt.subplots(1,1, figsize=(10,7))
			self.errors_df.hist(column='error', bins=int(np.sqrt(len(self.errors_df))),
							ax=ax, normed=True);
		except:
			raise ValueError('pipeline has not been fitted.')

	def get_feature_importance(self):
		
		try:
			if hasattr(self.pipeline, 'steps'):
				if hasattr(self.pipeline.steps[-1][-1], 'feature_importances_'):
					fig, ax = plt.subplots(1,1, figsize=(10,15))
					pd.DataFrame(self.pipeline.steps[-1][-1].feature_importances_, index=self.features_list
						).sort(0, ascending=True).plot.barh(ax=ax, fontsize=16)
				elif hasattr(self.pipeline.steps[-1][-1], 'coef_'):
					df = pd.DataFrame([self.pipeline.steps[-1][-1].intercept_] + 
						self.pipeline.steps[-1][-1].coef_.tolist(), index=['intercept'] + 
						self.features_list)
					print(df)
			else:
				fig, ax = plt.subplots(1,1, figsize=(10,15))
				pd.DataFrame(self.pipeline.feature_importances_, index=self.features_list
					).sort(0, ascending=True).plot.barh(ax=ax, fontsize=16)
		except:
			raise ValueError('pipeline has not been fitted.')

	def get_log_residuals(self):
		
		self.pipeline.fit(self.X, self.Y)
		Y_log = np.log1p(self.Y)
		Y_pred_log = np.log1p(self.pipeline.predict(self.X))
		residuals = Y_log - Y_pred_log
		residuals_df = pd.DataFrame({'residuals': residuals}, index=self.df.index)

		return residuals_df


class metamodel(object):
	def __init__(self, features, pipeline, n_folds, num_cv, cv_type, metric):
		self.features = features
		self.pipeline = pipeline
		self.n_folds = n_folds
		self.num_cv = num_cv
		self.cv_type = cv_type
		self.metric = metric

	def get_oos_cv_predictions(self, x_train, y_train, index_col):
		"""
		Method that generates the out-of-sample cv predictions and return model performance
		"""

		#Obtain features and output
		X = x_train[self.features].values
		index = x_train[index_col].values
		oos_predictions = np.zeros(len(index))

		# Perform #num_cv CV with different random seed to average predictions and reduce variance
		for random_number in range(self.num_cv):

			if hasattr(self.pipeline, 'predict_proba'):
				cv = StratifiedKFold(n_splits=self.n_folds, random_state= random_number, shuffle=True)
				cv_list = cv.split(X, y_train)
			else:
				if self.cv_type == 'group':
					gkf = GroupKFold(n_splits=self.n_folds)
					groups_for_cv = x_train[index_col].dt.date
					cv_list = list(gkf.split(x_train, y_train, groups_for_cv))
				else:
					cv = KFold(n_splits=self.n_folds, random_state= random_number, shuffle=True)
					cv_list = cv.split(X, y_train)

			print('CV number: ', random_number+1)

			#Check whether there are several error metrics
			if isinstance(self.metric, dict):
				CV_performance = { name : list() for name in self.metric.keys()}
			else:
				CV_performance = list()

			for fold, (train_index, test_index) in enumerate(cv_list):
				
				print('Acting on fold %i of' %(fold+1), random_number+1)

				self.pipeline.fit(X[train_index], y_train[train_index])
				
				if hasattr(self.pipeline, 'predict_proba'):
					y_pred = self.pipeline.predict_proba(X[test_index])[:,1]
				else:
					y_pred = self.pipeline.predict(X[test_index])

				if isinstance(self.metric, dict):
					for name, error_function in self.metric.items():
						CV_performance[name].append(error_function(y_train[test_index], y_pred))
				else:
					CV_performance.append(self.metric(y_train[test_index], y_pred))
				
				#Agregate prediction from each CV iteration
				oos_predictions[test_index] += y_pred

			if isinstance(CV_performance, dict):
				for name, val in CV_performance.items():
					print( str(name) + ' on %d CV: %0.4f +- %0.4f' %(random_number+1, np.mean(val), 2*np.std(val)))
			else:
				print('Model performance on %d CV : %0.4f +- %0.4f' %(random_number+1, np.mean(CV_performance), 2*np.std(CV_performance)))

		#Average each prediction
		oos_predictions = oos_predictions / self.num_cv

		return index, oos_predictions

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




