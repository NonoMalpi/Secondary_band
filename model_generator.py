import pandas as pd
import numpy as np
import datetime


class generate_model(object):
	def __init__(self, df_list, name_list):
		self.df_list = [self.__get_clean_data(df, name) for df, name in zip(df_list, name_list)]
	def __get_clean_data(self, df_original, name):
		weekday_dict = {
			0:'Wd', 1:'Wd', 2:'Wd', 3:'Wd', 4:'Wd', 5:'F', 6:'F' 
		}

		df = df_original.copy(deep=True)
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

		return clean_df_freq
	def return_df(self):
		return self.df_list