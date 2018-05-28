# Set up

%load_ext autoreload
%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from __future__ import absolute_import, division, print_function

import sys
import os

import pandas as pd
import numpy as np

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


pd.set_option('display.float_format', lambda x: '%.5f' % x)  # pandas
np.set_printoptions(precision=5, suppress=True)  # numpy

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# seaborn plotting style
sns.set(style='ticks', context='poster')

DATASETPATH = '/media/light/UbuntuDrive/Python_Code/Propython/Basic'


def load_DATA(bnppath=DATASETPATH):
    csv_path = os.path.join(
        DATASETPATH, "USCarrier_Traffic.csv")
    return pd.read_csv(csv_path, header=0, index_col=0, parse_dates=[0])


carrier = load_DATA()

carrier.head()
carrier.tail()

carrier.shape

# Create the training and testing series before analyzing the series
n_sample = carrier.shape[0]
n_train = int(0.95 * n_sample) + 1
n_forecast = n_sample - n_train

carrier_train = carrier.iloc[:n_train]['Total']
carrier_test = carrier.iloc[n_train:]['Total']

print(carrier_train.shape)
print(carrier_test.shape)

carrier_train.head()

# This function is for plooting the ACF and PACF


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


tsplot(carrier_train, title='US-Carrier', lags=20)

# Throgh the plot we can clearly see the positive trend exists

log_carrier_total = np.log(carrier_train)
log_carrier_diff = log_carrier_total.diff()
log_carrier_diff = log_carrier_diff.dropna()


tsplot(log_carrier_total, title='Natural-Log-Carrier', lags=20)

tsplot(log_carrier_diff, title='Log-diff-Carrier', lags=40)

# we can see here the 1st value the acf first crosses the 95% conf interval so AR(1)
# we can see the PACF crosses first time around 95% conf interval MA(1)
# as we took diif as 1 and make the residual enforce_stationarity
# Also we can clearly see from the autocorrelation the seasonality exists with 1 year 2 year 3 year

# checking which particular month the sales are high

carrier['period'] = carrier.index.strftime('%b')
carrier['Year'] = carrier.index.year

carrier_piv = carrier.pivot(index='Year', columns='period', values='Total')

carrier = carrier.drop(['period', 'Year'], axis=1)

# put the months in order
month_names = pd.date_range(
    start='2016-01-01', periods=12, freq='MS').strftime('%b')
carrier_piv = carrier_piv.reindex(columns=month_names)

# plot it
fig, ax = plt.subplots(figsize=(8, 6))
carrier_piv.plot(ax=ax, kind='box')

ax.set_xlabel('Month')
ax.set_ylabel('Total')
ax.set_title('Boxplot of seasonal values')
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout()

# As with this figure we are certailnly sure that during MAY , JUNE , JULY the Total is high

# This function is useful to run the stats


def model_resid_stats(model_results,
                      het_method='breakvar',
                      norm_method='jarquebera',
                      sercor_method='ljungbox',
                      verbose=True,
                      ):
    '''More information about the statistics under the ARIMA parameters table, tests of standardized residuals:

    Test of heteroskedasticity
    http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_heteroskedasticity.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_heteroskedasticity

    Test of normality (Default: Jarque-Bera)
    http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_normality.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_normality

    Test of serial correlation (Default: Ljung-Box)
    http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_serial_correlation.html
    '''
    # Re-run the ARIMA model statistical tests, and more. To be used when selecting viable models.
    (het_stat, het_p) = model_results.test_heteroskedasticity(het_method)[0]
    norm_stat, norm_p, skew, kurtosis = model_results.test_normality(norm_method)[
        0]
    sercor_stat, sercor_p = model_results.test_serial_correlation(
        method=sercor_method)[0]
    sercor_stat = sercor_stat[-1]  # last number for the largest lag
    sercor_p = sercor_p[-1]  # last number for the largest lag

    # Run Durbin-Watson test on the standardized residuals.
    # The statistic is approximately equal to 2*(1-r), where r is the sample autocorrelation of the residuals.
    # Thus, for r == 0, indicating no serial correlation, the test statistic equals 2.
    # This statistic will always be between 0 and 4. The closer to 0 the statistic,
    # the more evidence for positive serial correlation. The closer to 4,
    # the more evidence for negative serial correlation.
    # Essentially, below 1 or above 3 is bad.
    dw_stat = sm.stats.stattools.durbin_watson(
        model_results.filter_results.standardized_forecasts_error[0, model_results.loglikelihood_burn:])

    # check whether roots are outside the unit circle (we want them to be);
    # will be True when AR is not used (i.e., AR order = 0)
    arroots_outside_unit_circle = np.all(np.abs(model_results.arroots) > 1)
    # will be True when MA is not used (i.e., MA order = 0)
    maroots_outside_unit_circle = np.all(np.abs(model_results.maroots) > 1)

    if verbose:
        print('Test heteroskedasticity of residuals ({}): stat={:.3f}, p={:.3f}'.format(
            het_method, het_stat, het_p))
        print('\nTest normality of residuals ({}): stat={:.3f}, p={:.3f}'.format(
            norm_method, norm_stat, norm_p))
        print('\nTest serial correlation of residuals ({}): stat={:.3f}, p={:.3f}'.format(
            sercor_method, sercor_stat, sercor_p))
        print(
            '\nDurbin-Watson test on residuals: d={:.2f}\n\t(NB: 2 means no serial correlation, 0=pos, 4=neg)'.format(dw_stat))
        print('\nTest for all AR roots outside unit circle (>1): {}'.format(
            arroots_outside_unit_circle))
        print('\nTest for all MA roots outside unit circle (>1): {}'.format(
            maroots_outside_unit_circle))

    stat = {'het_method': het_method,
            'het_stat': het_stat,
            'het_p': het_p,
            'norm_method': norm_method,
            'norm_stat': norm_stat,
            'norm_p': norm_p,
            'skew': skew,
            'kurtosis': kurtosis,
            'sercor_method': sercor_method,
            'sercor_stat': sercor_stat,
            'sercor_p': sercor_p,
            'dw_stat': dw_stat,
            'arroots_outside_unit_circle': arroots_outside_unit_circle,
            'maroots_outside_unit_circle': maroots_outside_unit_circle,
            }
    return stat

# This particular method is used to find out the best parameter for arima and seasnality


def model_gridsearch(ts,
                     p_min,
                     d_min,
                     q_min,
                     p_max,
                     d_max,
                     q_max,
                     sP_min,
                     sD_min,
                     sQ_min,
                     sP_max,
                     sD_max,
                     sQ_max,
                     trends,
                     s=None,
                     enforce_stationarity=True,
                     enforce_invertibility=True,
                     simple_differencing=False,
                     plot_diagnostics=False,
                     verbose=False,
                     filter_warnings=True,
                     ):
    '''Run grid search of SARIMAX models and save results.
    '''

    cols = ['p', 'd', 'q', 'sP', 'sD', 'sQ', 's', 'trend',
            'enforce_stationarity', 'enforce_invertibility', 'simple_differencing',
            'aic', 'bic',
            'het_p', 'norm_p', 'sercor_p', 'dw_stat',
            'arroots_gt_1', 'maroots_gt_1',
            'datetime_run']

    # Initialize a DataFrame to store the results
    df_results = pd.DataFrame(columns=cols)

    # # Initialize a DataFrame to store the results
    # results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
    #                            columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

    mod_num = 0
    for trend, p, d, q, sP, sD, sQ in itertools.product(trends,
                                                        range(
                                                            p_min, p_max + 1),
                                                        range(
                                                            d_min, d_max + 1),
                                                        range(
                                                            q_min, q_max + 1),
                                                        range(
                                                            sP_min, sP_max + 1),
                                                        range(
                                                            sD_min, sD_max + 1),
                                                        range(
                                                            sQ_min, sQ_max + 1),
                                                        ):
        # initialize to store results for this parameter set
        this_model = pd.DataFrame(index=[mod_num], columns=cols)

        if p == 0 and d == 0 and q == 0:
            continue

        try:
            model = sm.tsa.SARIMAX(ts,
                                   trend=trend,
                                   order=(p, d, q),
                                   seasonal_order=(sP, sD, sQ, s),
                                   enforce_stationarity=enforce_stationarity,
                                   enforce_invertibility=enforce_invertibility,
                                   simple_differencing=simple_differencing,
                                   )

            if filter_warnings is True:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model_results = model.fit(disp=0)
            else:
                model_results = model.fit()

            if verbose:
                print(model_results.summary())

            if plot_diagnostics:
                model_results.plot_diagnostics()

            stat = model_resid_stats(model_results,
                                     verbose=verbose)

            this_model.loc[mod_num, 'p'] = p
            this_model.loc[mod_num, 'd'] = d
            this_model.loc[mod_num, 'q'] = q
            this_model.loc[mod_num, 'sP'] = sP
            this_model.loc[mod_num, 'sD'] = sD
            this_model.loc[mod_num, 'sQ'] = sQ
            this_model.loc[mod_num, 's'] = s
            this_model.loc[mod_num, 'trend'] = trend
            this_model.loc[mod_num,
                           'enforce_stationarity'] = enforce_stationarity
            this_model.loc[mod_num,
                           'enforce_invertibility'] = enforce_invertibility
            this_model.loc[mod_num,
                           'simple_differencing'] = simple_differencing

            this_model.loc[mod_num, 'aic'] = model_results.aic
            this_model.loc[mod_num, 'bic'] = model_results.bic

            # this_model.loc[mod_num, 'het_method'] = stat['het_method']
            # this_model.loc[mod_num, 'het_stat'] = stat['het_stat']
            this_model.loc[mod_num, 'het_p'] = stat['het_p']
            # this_model.loc[mod_num, 'norm_method'] = stat['norm_method']
            # this_model.loc[mod_num, 'norm_stat'] = stat['norm_stat']
            this_model.loc[mod_num, 'norm_p'] = stat['norm_p']
            # this_model.loc[mod_num, 'skew'] = stat['skew']
            # this_model.loc[mod_num, 'kurtosis'] = stat['kurtosis']
            # this_model.loc[mod_num, 'sercor_method'] = stat['sercor_method']
            # this_model.loc[mod_num, 'sercor_stat'] = stat['sercor_stat']
            this_model.loc[mod_num, 'sercor_p'] = stat['sercor_p']
            this_model.loc[mod_num, 'dw_stat'] = stat['dw_stat']
            this_model.loc[mod_num,
                           'arroots_gt_1'] = stat['arroots_outside_unit_circle']
            this_model.loc[mod_num,
                           'maroots_gt_1'] = stat['maroots_outside_unit_circle']

            this_model.loc[mod_num, 'datetime_run'] = pd.to_datetime(
                'today').strftime('%Y-%m-%d %H:%M:%S')

            df_results = df_results.append(this_model)
            mod_num += 1
        except:
            continue
    return df_results


# run model grid search

p_min = 0
d_min = 0
q_min = 0
p_max = 6
d_max = 1
q_max = 6

sP_min = 0
sD_min = 0
sQ_min = 0
sP_max = 1
sD_max = 1
sQ_max = 1

s = 12

# trends=['n', 'c']
trends = ['n']

enforce_stationarity = True
enforce_invertibility = True
simple_differencing = False

plot_diagnostics = False

verbose = True
df_results = model_gridsearch(log_carrier_total,
                              p_min,
                              d_min,
                              q_min,
                              p_max,
                              d_max,
                              q_max,
                              sP_min,
                              sD_min,
                              sQ_min,
                              sP_max,
                              sD_max,
                              sQ_max,
                              trends,
                              s=s,
                              enforce_stationarity=enforce_stationarity,
                              enforce_invertibility=enforce_invertibility,
                              simple_differencing=simple_differencing,
                              plot_diagnostics=plot_diagnostics,
                              verbose=verbose,
                              )


df_results.sort_values(by='bic').head(10)

# as with this table we can see that p,d,q = 1,1,1 and sp,sd,sq = 1,0,1 with lowest bic score as well as less than 0.05 nomralized p values
# we should take this and build the models

mod = sm.tsa.statespace.SARIMAX(log_carrier_total, order=(
    1, 1, 1), seasonal_order=(1, 0, 1, 12), simple_differencing=False)

sarima_fit2 = mod.fit()
print(sarima_fit2.summary())

sarima_fit2.plot_diagnostics()


# transforming the predicted value actual values and compare

forecast = sarima_fit2.predict(start=13, end=180)

forecast_outofbag = forecast[-9:]

forecast_outofbag

actual_forecast = np.exp(forecast_outofbag)

actual_forecast
carrier_test


rms = sqrt(mean_squared_error(carrier_test, actual_forecast))

rms

# Now plotting the actual and forecasted values
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
ax1.plot(carrier_test, label='Held-out data', linestyle='--')
ax1.plot(actual_forecast, 'r', label='forecasted values')
