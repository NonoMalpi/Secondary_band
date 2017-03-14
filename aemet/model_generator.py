import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


class generate_df(object):
	
	def __init__(self, path_dict, main_name):
		self.raw_df_list = { name : self.__import_files(path_df) for name, path_df in path_dict.items()}
		self.df_dict = { name : self.__get_clean_data(df, name) for name, df in self.raw_df_list.items()}
		self.name_list = list(path_dict.keys())
		self.main_name = main_name
		self.completed_df = self.__return_merged_df()

	def __import_files(self, path):
		df = pd.read_csv(path, encoding='latin1', delimiter=';')
		return df
	
	def __get_clean_data(self, df_original, name):
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
		df = df[df.index < '2017']
		clean_df = df[['date', 'year', 'month', 'season', 'day','weekday','time', 'hour', 'minute', 'value']]
		clean_df = clean_df[~clean_df.index.duplicated()]
		clean_df['hour'] = np.where(clean_df['hour'].isin(np.arange(9,23)), 'Peak', 'off_peak')
		clean_df = clean_df.rename(columns={'value':name})
		clean_df_freq = clean_df.asfreq('H')
		#if name in ['france', 'morocco', 'portugal']:
		#	clean_df_freq[name] = np.where(clean_df_freq[name].isnull(), 0, clean_df_freq[name])

		return clean_df_freq

	def __return_merged_df(self):
		"""
		Return a merged df with all features, base_df refers to the name of the df
		to use as base of the merged df
		"""
		return self.df_dict[self.main_name].join([self.df_dict[name][[name]] for name in self.name_list if name != self.main_name])

	def return_raw_df_dict(self):
		return self.raw_df_list

	def return_df_dict(self):
		return self.df_dict

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

		for feature in name_list_24:
			df[feature + '-24'] = df[feature].shift(periods=24)

		df.dropna(subset=[self.main_name], inplace=True)

		return df

class train_model(object):

	def __init__(self, df, features, output, features_to_remove, n_folds):
		self.df = df
		self.features = features
		self.output = output
		self.features_to_remove = features_to_remove
		self.n_folds = n_folds
		self.X, self.Y = self.__divide_features_output()
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
																train_size=0.8, test_size=0.2, 
																random_state=0)

	def __divide_features_output(self):
		X = self.df[self.features].values
		Y = self.df[self.output].values
		return X, Y

	def obtain_cv_score(self, pipeline):
		kf = KFold(n_splits=self.n_folds, random_state=0)

		CV_mse = list()
		CV_mae = list()

		for train_index, test_index in kf.split(self.x_train, self.y_train):
			pipeline.fit(self.x_train[train_index], self.y_train[train_index])
			output_pred = pipeline.predict(self.x_train[test_index])
			CV_mae.append(mean_absolute_error(self.y_train[test_index], output_pred))
			CV_mse.append(mean_squared_error(self.y_train[test_index], output_pred))

		print('Mean absolute error: %0.4f +- %0.4f' %(np.mean(CV_mae), 2*np.std(CV_mae)))
		print('Mean squared error: %0.4f +- %0.4f' %(np.mean(CV_mse), 2*np.std(CV_mse)))

	def obtain_train_test_error(self, pipeline):
		pipeline.fit(self.x_train, self.y_train)
		
		print('Train MAE: ' +str(mean_absolute_error(self.y_train, pipeline.predict(self.x_train))) +
			  ', Train MSE: ' +str(mean_squared_error(self.y_train, pipeline.predict(self.x_train))))

		print('Test MAE: ' +str(mean_absolute_error(self.y_test, pipeline.predict(self.x_test))) +
			  ', Test MSE: ' +str(mean_squared_error(self.y_test, pipeline.predict(self.x_test))))

		errors_df = pd.DataFrame({'y_true':self.y_test, 'y_pred':pipeline.predict(self.x_test)})
		errors_df['error'] = errors_df['y_true'] - errors_df['y_pred']

		self.pipeline = pipeline
		self.errors_df = errors_df

	def plot_histogram_error(self):
		fig, ax = plt.subplots(1,1, figsize=(10,7))
		self.errors_df.hist(column='error', bins=int(np.sqrt(len(self.errors_df))),
							ax=ax, normed=True);

	def plot_feature_importance(self):
		fig, ax = plt.subplots(1,1, figsize=(10,15))
		pd.DataFrame(self.pipeline.feature_importances_, index=self.df.drop(labels=self.features_to_remove
					+ [self.output], axis=1).columns.tolist()).sort(0, ascending=True).plot.barh(ax=ax, fontsize=16)

class metamodel(object):
	def __init__(self, features, pipeline, n_folds, num_cv, metric):
		self.features = features
		self.pipeline = pipeline
		self.n_folds = n_folds
		self.num_cv = num_cv
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
			else:
				cv = KFold(n_splits=self.n_folds, random_state= random_number, shuffle=True)

			print('CV number: ', random_number+1)

			#Check whether there are several error metrics
			if isinstance(self.metric, dict):
				CV_performance = { name : list() for name in self.metric.keys()}
			else:
				CV_performance = list()

			for fold, (train_index, test_index) in enumerate(cv.split(X, y_train)):
				
				print('Acting in fold %i on' %(fold+1), random_number+1)

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