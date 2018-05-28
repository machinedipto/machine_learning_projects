import findspark
findspark.init('/media/light/UbuntuDrive/work/spark-2.3.0-bin-hadoop2.7')

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt

# Set up

%load_ext autoreload
%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from __future__ import absolute_import, division, print_function

import sys
import os

# Remote Data Access
import datetime
# reference: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html

# TSA from Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

import itertools
import warnings

from statsmodels.graphics.api import qqplot

# Display and Plotting
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from math import sqrt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA


spark = SparkSession.builder.master(
    'local').appName('stock_analysis').getOrCreate()

data = spark.read.csv(
    '/home/light/Downloads/BHARTIARTLALLN.csv', inferSchema=True, header=True)

data.show()

data.printSchema()

time_data = data.select(['Date', 'Close Price'])

time_data_df = time_data.toPandas()

time_data_df['Date'] = pd.to_datetime(time_data_df['Date'], unit='D')

time_data_df.dtypes

time_data_df

time_data_df = time_data_df.rename(columns={'Close Price': 'Close_Price'})


def bbands(price, length=20, numsd=2):
    """ returns average, upper band, and lower band"""
    ave = pd.stats.moments.rolling_mean(price, length)
    sd = pd.stats.moments.rolling_std(price, length)
    upband = ave + (sd * numsd)
    dnband = ave - (sd * numsd)
    return np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)


time_data_df['ave'], time_data_df['upper'], time_data_df['lower'] = bbands(
    time_data_df.Close_Price, length=20, numsd=2)

time_data_df1 = time_data_df[19:]

time_data_df1

time_data_df1.plot(y=['Close_Price', 'ave', 'upper',
                      'lower'], title='Bollinger Bands')

# Question 2

close_list = list(time_data_df1.Close_Price)
upper_list = list(time_data_df1.upper)
lower_list = list(time_data_df1.lower)

count = 0


def countvioleted(close_list, upper_list, lower_list):
    count = 0
    for i in range(len(close_list)):
        if close_list[i] > upper_list[i] or close_list[i] < lower_list[i]:
            count = count + 1
    return count


violation = countvioleted(close_list, upper_list, lower_list)

print(violation)

# 3rd question is fuck all so fuck you

DATASETPATH = '/home/light/Downloads/BHARTIARTLALLN.csv'

time_data_df = pd.read_csv(DATASETPATH, header=0,
                           parse_dates=True, index_col=2)

time_data_df

pd.set_option('display.float_format', lambda x: '%.5f' % x)  # pandas
np.set_printoptions(precision=5, suppress=True)  # numpy

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

time_data_df

# seaborn plotting style
sns.set(style='ticks', context='poster')

n_smaple = time_data_df.shape[0]
n_train = int(n_smaple * .70) + 1

n_smaple
n_forecast = n_smaple - n_train

time_data_df = time_data_df.rename(columns={'Close Price': 'Close_Price'})


original_data = time_data_df.fillna(method='ffill')

original_data.shape
original_data

time_data_df

start_date = '2017-01-02'
end_date = '2017-12-31'

ran = pd.date_range(start_date, end_date, freq='D')

ran

time_data_df['Close_Price']

original_data.index = ran

original_data = pd.Series(original_data['Close_Price'], index=ran)

original_data = original_data.fillna(method='ffill')
type(time_data_df)

original_data

time_data_df
#time_data_df = time_data_df.set_index('Date')
time_data_df

time_data_df_train = time_data_df.iloc[:n_train]['Close_Price']
time_data_df_test = time_data_df.iloc[n_train:]['Close_Price']

type(time_data_df_train)


def tsplot(y, lags=None, title='', figsize=(14, 8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    fig.tight_layout()
    return ts_ax, acf_ax, pacf_ax


tsplot(time_data_df_train, title='stock_data', lags=20)

# we can see some trend exists we need to difference


log_stock = np.log(time_data_df_train)
log_diff_stock = log_stock.diff(2)
log_diff_stock = log_diff_stock.dropna()

type(log_stock)

log_stock

tsplot(log_stock, title='Natural-Log-stock', lags=20)

tsplot(log_diff_stock, title='Log-diff-stock', lags=40)

###
result = adfuller(log_diff_stock)

result[1]

# from the graph we can see there are spike at ACF value 2 so the q in arima model or MA value in ARIMA model is 2

# from the graph we can see that there are PACF value is 2 now the actual AR value for the data is 2

# SO in ARIMA model i m going to apply 2,0,2 models


log_stock.index

mod = ARIMA(log_stock, order=(
    3, 2, 1))

arima_fit = mod.fit()


print(arima_fit.summary())


log_stock


# transforming the predicted value actual values and compare

len(log_stock)

log_stock.values

mod.nobs

n_forecast = 20

time_data_df_test


forecast = arima_fit.predict(start='2017-01-04', end='2017-12-29')


forecast


forecast_outofbag = forecast[-73:]

forecast_outofbag

actual_forecast = np.exp()

actual_forecast
time_data_df_test


rms = sqrt(mean_squared_error(carrier_test, actual_forecast))

rms

# Now plotting the actual and forecasted values
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
ax1.plot(carrier_test, label='Held-out data', linestyle='--')
ax1.plot(actual_forecast, 'r', label='forecasted values')
