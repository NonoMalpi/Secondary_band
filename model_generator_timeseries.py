import datetime
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error


class generate_df(object):
    """Generate a `pandas.DataFrame` to fit the ML models.
    This class allows to generate a DataFrame from several csv files; these
    files contain the feature values of the Spanish electricity system. 
    In addition, a feature engineering steps over the raw variables is inlcuded.

    Attributes
    ----------
    name_list: list
        List containing the name of all features used.

    main_name: str
        Name of the target variable for teh ML model.

    completed_df: pandas.DataFrame
        DataFrame with target variable, raw and time-dependent fetures. 
    """

    def __init__(self, path_dict, main_name, date):
        """
        Parameters
        ----------
        path_dict: dict
            Dictionary containing the name of the feature and the path.

        main_name: str
            Name of the target variable for teh ML model.

        date: str
            The last date for which data is available.
        """
        # dict containing name feature and raw df
        self.__raw_df_list = { 
            name : self.__import_files(path_df) 
            for name, path_df in path_dict.items()
        }

        # dict containing clean df until the date passed 
        self.__df_dict = { 
            name : self.__get_clean_data(df, name, date) 
            for name, df in self.__raw_df_list.items()
        }
        
        self.name_list = list(path_dict.keys())
        self.main_name = main_name

        # df with all target variable, time-dependent columns and raw features
        self.completed_df = self.__return_merged_df()

    def __import_files(self, path):
        # parse csv file into pandas.DataFrame
        df = pd.read_csv(path, encoding='latin1', delimiter=';')
        return df
    
    def __get_clean_data(self, df_original, name, date):
        """
        Clean raw df and add time-dependent features from each timestamp.

        Parameters
        ----------
        df_original: pandas.DataFrame
            Raw DataFrame containig the feature value and the timestamp.

        name: str
            Feature name to use.

        date: str
            Last date to consider.

        Returns
        -------
        clean_df: pandas.DataFrame
            DataFrame containing the feature value together with the 
            time-dependent features extracted.
        """
        # dict mapping weekday or fest
        weekday_dict = {
            0:'Wd', 1:'Wd', 2:'Wd', 3:'Wd', 4:'Wd', 5:'F', 6:'F' 
        }

        df = df_original.copy(deep=True)

        # select Spain ('geoid' == 3) in spot file 
        if name == 'spot':
            df = df[df['geoid'] == 3]

        # clean date and extract features from 'datetime' timestamp
        # since the timestamp is UTC+1, it has to be parser without the +1
        df.loc[:, 'date'] =  pd.to_datetime(
            df['datetime'].apply(lambda x: x[:10]), format='%Y-%m-%d'
        )

        df.loc[:, 'year'] = df['date'].dt.year
        
        df.loc[:, 'time'] = pd.to_datetime(
            df['datetime'].apply(lambda x: x[:19])
        ).dt.time
        
        df.loc[:, 'month'] = df['date'].dt.month
        
        df.loc[:, 'day'] = df['date'].dt.day
        
        df.loc[:, 'hour'] = df['datetime'].apply(lambda x: x[11:13]).astype(int)
        
        df.loc[:, 'weekday'] = df['date'].dt.dayofweek
        df.replace({'weekday':weekday_dict}, inplace=True)

        df.loc[:, 'season'] = np.where(
            df['month'].isin(list(range(4,10))), 'summer', 'winter'
        )

        # create index combaining date and time
        df.loc[:, 'date_hour'] = df.apply(
            lambda x: datetime.datetime.combine(x['date'], x['time']), axis=1
        )
        df.set_index('date_hour', inplace=True)
        df = df[df.index < date]
        
        # keep just desired features
        clean_df = df[[
            'date', 'year', 'month', 
            'season', 'day','weekday',
            'time', 'hour', 'value'
        ]]
        #  remove duplicated hours (when the time is delayed one hour)
        clean_df = clean_df[~clean_df.index.duplicated()]
        
        # classify hour according to peak demand
        clean_df.loc[:, 'hour'] = np.where(
            clean_df['hour'].isin(np.arange(9,23)), 'Peak', 'off_peak'
        )

        clean_df = clean_df.rename(columns={'value':name})

        return clean_df

    def __return_merged_df(self):
        """
        Return a merged df from the target df (output variable 
        and time-depend features) joining the raw features columns
        """
        return self.__df_dict[self.main_name].join(
            [self.__df_dict[name][[name]] for name in self.name_list 
            if name != self.main_name]
        )

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
        """
        Generate a DataFrame containing feature engineered from raw variables.
        The feature engineering contains:
            - One hot encoding for the hour, weekday and season.
            - Values from the last hour, when available.
            - Values from the last day at the same hour.
            - Moving average of the last 24 hours values.

        Parameters
        ----------
        name_list_24: list
            List containing the name of the variables for which the values of 
            the last hour or the moving average cannot be performed.

        Returns
        -------
        df: pandas.Dataframe
            DataFrame ready to train the model (raw variables, 
            feature engineering and target output)

        """
        df = self.completed_df[self.name_list + ['hour', 'weekday','season']]
        
        # one hot encoding of hour, weekday and season
        for feature in ['hour', 'weekday', 'season']:
            df[
                pd.get_dummies(df[feature], drop_first=True).columns.tolist()
            ] = pd.get_dummies(df[feature], drop_first=True)

        # create a list with teh features to apply all feature engineering steps
        custom_list_1_24 = [
            feature for feature in self.name_list 
            if feature not in list(set([self.main_name] + name_list_24))
        ]
        # apply transformations
        for feature in custom_list_1_24:
            df.loc[:, feature + '-1'] = df[feature].shift(periods=1)
            df.loc[:, feature + '-24'] = df[feature].shift(periods=24)
            df.loc[:, feature + '_mov_aver_24'] = pd.rolling_mean(
                df[feature], window=24
            )
        # apply only possible transformation for features in name_list_24
        for feature in name_list_24:
            df.loc[:, feature + '-24'] = df[feature].shift(periods=24)

        # remove rows with NaN in target output
        df.dropna(subset=[self.main_name], inplace=True)

        return df


class timeseries_model(object):
    """Generate a ML model for timeseries that can be trained and tested with custom cv.
    This class allows to instantiate a ML model, trained and tested it on a 
    rolling basis cv, check the errors of the model and get the feature 
    importance if available.

    Attributes
    ----------
    df: pandas.DataFrame
        Input DataFrame filtered according to the first date to 
        consider train set.

    features_list: list
        List containing the name of the input features for X.

    output: str
        Name of the targte variable Y for the ML model.

    date_index: pandas.DatetimeIndex
        DatetimeIndex object from start and end date on a daily basis.

    length: int
        Number of days used to train the model if the X input is
        a rolling window.

    pipeline: sklearn.pipeline.Pipeline
        Pipeline of transformers with final estimators.

    date_to_predict: str
        First date to predict.

    rolling_bool: bool
        Boolean flag to indicate if the ML model is trained with 
        a rolling window.

    cv: list
        Sequential list containing a list of the numerical indexes 
        of the train and test set.

    X: numpy array of shape [n_samples, n_features]
        Training set.
    
    Y: numpy array of shape [n_samples]
        Target values.

    CV_mae, CV_mse: list
        List of daily mean absolute and squared error for 
        out-of-samples predictions.

    result_df: pandas.DataFrame
        DataFrame containinng the target and prediciton values.

    fitted_residuals_std: list
        Daily standard deviation of the residuals.
    """

    def __init__(
            self, df, features_list, output, start_date, end_date, 
            train_length, pipeline, first_date_to_predict='2017-01-01',
            rolling=False
        ):
        """
        Parameters
        ----------
        df: pandas.DataFrame
            Timeseries input df for the ML model.

        features_list: list
            List containing the name of the input features for X.

        output: str
            Name of the target variable Y for the ML model.

        start_date: str, dateformat: '%Y-%m-%d'
            Minimum date used to conform X.

        end_date: str, dateformat: '%Y-%m-%d'
            Maximum date used to conform X.

        train_length: int
            Number of days used to train the model if the X input is 
            a rolling window.

        pipeline: sklearn.pipeline.Pipeline
            Pipeline of transformers with final estimator.

        first_date_to_predict: str, dateformat: '%Y-%m-%d', default: '2017-01-01'
            Date after which the model will start yielding predictions.

        rolling: bool, default: False.
            This controls if the model is trained on a rolling window basis.
        """

        self.df = self.__get_df(df, start_date)
        self.features_list = features_list
        self.output = output
        self.date_index = self.__get_date_index(start_date, end_date)
        self.length = train_length
        self.pipeline = pipeline
        self.date_to_predict = first_date_to_predict
        self.rolling_bool = rolling

        # create custom cv keeping indexes of train and test.
        self.cv = self.__create_cv_indexes()

        # split df into X and Y.
        self.X, self.Y = self.__divide_features_output()

    def __get_df(self, df, start_date):
        """
        Filter the input df from the start date  
        """
        df = df[df.index >= start_date]
        return df

    def __get_date_index(self, start_date, end_date):
        """
        Generate a DatetimeIndex object from start and end date on a daily basis.
        """
        date_index = pd.date_range(start=start_date, end=end_date, freq='D')
        return date_index

    def __create_cv_indexes(self):
        """
        Create a custom cv depending on if the training method is on a 
        rolling basis.

        Returns
        -------
        cv: list
            Sequential list containing a list of the numerical indexes 
            of the train and test set.
        """
        cv_list = list()

        # keep the numerical index of the first date to predict and subtract 
        # the length of the training window. 
        first_date = np.flatnonzero(
            self.date_index == self.date_to_predict
        )[0] - self.length

        # from that first date to predict, generate the indexes of the 
        # train and test sets until the last day to predcit.
        for i in range(first_date, len(self.date_index) - self.length):
            # if rolling_bool is set as True, the training set is built just
            # from the last self.length days, otherwise the training set is 
            # built with all previous availbale days.
            train_period = self.date_index.date[i*self.rolling_bool:self.length+i]
            test_period = self.date_index.date[self.length+i]

            # keep the numerical indexes for train and test set and add to list
            train_index = self.df.reset_index()[
                self.df.reset_index()['date_hour'].dt.date.isin(train_period)
            ].index

            test_index = self.df.reset_index()[
                self.df.reset_index()['date_hour'].dt.date == (test_period)
            ].index

            # add to list and append to sequential cv.
            train_test_list = [train_index, test_index]
            cv_list.append(train_test_list)
        
        return cv_list

    def __divide_features_output(self):
        X = self.df[self.features_list].values
        Y = self.df[self.output].values
        return X, Y

    def obtain_cv_scores(self):
        """
        Generate out-of-sample predictions, calculate standrard deviation of 
        in-sample residuals and store predictions and the associated error.
        """
        # define list of metrics, residuals std dev and output.
        CV_mae = list()
        CV_mse = list()
        fitted_residuals_std = list()
        output_list = list()

        # using the attribute cv of the class, train and test on a rolling basis.
        for fold, (train_index, test_index) in enumerate(self.cv):
            if ((fold+1) % 10) == 0:
                print('Acting on fold %i' %(fold+1))

            # fit until the corresponding day.
            self.pipeline.fit(self.X[train_index], self.Y[train_index])

            # compute the standard deviation of the residuals using 
            # log transformation.
            fitted_residuals_std.append(
                np.std(np.log1p(self.Y[train_index]) - \
                np.log1p(self.pipeline.predict(self.X[train_index])))
            )

            # yield prediciton for test day.
            output_pred = self.pipeline.predict(self.X[test_index])

            # store mean absolute and squared error for test day.
            CV_mae.append(mean_absolute_error(self.Y[test_index], output_pred))
            CV_mse.append(mean_squared_error(self.Y[test_index], output_pred))

            output_list.append(output_pred)

        # concat all out-of-sample predicitons
        output = np.concatenate(output_list)

        print('Mean absolute error: %0.4f +- %0.4f' \
            %(np.mean(CV_mae), 2*np.std(CV_mae))
        )
        print('Mean squared error: %0.4f +- %0.4f' \
            %(np.mean(CV_mse), 2*np.std(CV_mse))
        )

        self.CV_mae = CV_mae
        self.CV_mse = CV_mse

        # store true value and oos prediction in result_df attribute
        first_date = np.flatnonzero(self.df.index >= self.date_to_predict)[0]

        self.result_df = pd.DataFrame(
            {'y_true':self.Y[first_date:], 'y_pred':output},
            index=self.df.iloc[first_date:].index
        )
        self.result_df['error'] = self.result_df['y_true'] - \
                                  self.result_df['y_pred']

        self.fitted_residuals_std = fitted_residuals_std

    def plot_histogram_error(self):
        """
        Plot histogram of errors for the period predicted
        """
        fig, ax = plt.subplots(1,1, figsize=(10,7))
        self.result_df.hist(
            column='error', 
            bins=int(np.sqrt(len(self.result_df))), 
            ax=ax,
            normed=True
        )
        ax.grid(linestyle='--', linewidth=1, color='gray', alpha=0.35)

    def get_feature_importance(self):
        """
        Obtain the feature importance for those models that have the possibility.
        """
        try:
            if hasattr(self.pipeline, 'steps'):

                if hasattr(self.pipeline.steps[-1][-1], 'feature_importances_'):
                    # pipeline with steps and estimator has attribute 
                    # feature_importances
                    fig, ax = plt.subplots(1,1, figsize=(10,20))
                    pd.DataFrame(
                        self.pipeline.steps[-1][-1].feature_importances_, 
                        index=self.features_list
                    ).sort_values(0, ascending=True).plot.barh(
                        ax=ax, fontsize=16
                    )

                elif hasattr(self.pipeline.steps[-1][-1], 'coef_'):

                    if hasattr(self.pipeline.steps[-2][-1], 'get_support'):
                        # pipeline with steps and estimators is parametric
                        # and it also has an intermediate step with feature 
                        # selection

                        select_feat = self.pipeline.steps[-2][-1].get_support()
                        df = pd.DataFrame(
                            [self.pipeline.steps[-1][-1].intercept_] + 
                            self.pipeline.steps[-1][-1].coef_.tolist(), 
                            index=['intercept'] + \
                            list(compress(self.features_list, select_feat))
                        )
                        print(df)

                    else:
                        # pipeline with steps and estimators is parametric
                        df = pd.DataFrame(
                            [self.pipeline.steps[-1][-1].intercept_] + 
                            self.pipeline.steps[-1][-1].coef_.tolist(), 
                            index=['intercept'] + self.features_list)
                        print(df)

            else:
                # pipeline without steps
                fig, ax = plt.subplots(1,1, figsize=(10,20))
                pd.DataFrame(
                    self.pipeline.feature_importances_, 
                    index=self.features_list
                ).sort_values(0, ascending=True).plot.barh(ax=ax, fontsize=16)
        except:

            raise ValueError('pipeline has not been fitted.')
