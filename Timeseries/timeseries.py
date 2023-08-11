import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot, lag_plot
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from scipy import signal
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_df(x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


def decomposition(df, model):
    if model == 'multiplicative':
        # Multiplicative Decomposition
        result = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')
    elif model == 'additive':
        # Additive Decomposition
        result = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')
    else:
        result = None
    return result


def is_stationary_adf(df_values):
    # ADF Test
    result = adfuller(df_values, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
    return result


def reconstruct_decomposed(result):
    df_reconstructed = pd.concat([result.seasonal, result.trend, result.resid, result.observed], axis=1)
    return df_reconstructed


def detrend_substract_least_squares_fit(df_values):
    detrended = signal.detrend(df_values)
    return detrended


def detrend_substract_trend(df_values):
    result_mul = seasonal_decompose(df_values, model='multiplicative', extrapolate_trend='freq')
    detrended = df_values - result_mul.trend
    return detrended


def deseasonalize(df_values):
    # Time Series Decomposition
    result_mul = seasonal_decompose(df_values, model='multiplicative', extrapolate_trend='freq')
    # Deseasonalize
    deseasonalized = df_values / result_mul.seasonal
    return deseasonalized


def detect_seasonality(df_values):
    autocorrelation_plot(df_values.tolist())
    plt.show()


def backward_fill(df):
    df_bfill = df.bfill()
    return df_bfill


def forward_fill(df):
    df_ffill = df.ffill()
    return df_ffill


def linear_interpollation_fill(df):
    df['rownum'] = np.arange(df.shape[0])
    df_nona = df.dropna(subset=['value'])
    f = interp1d(df_nona['rownum'], df_nona['value'])
    df['linear_fill'] = f(df['rownum'])
    return df


def cubic_interpollation_fill(df):
    df['rownum'] = np.arange(df.shape[0])
    df_nona = df.dropna(subset=['value'])
    f = interp1d(df_nona['rownum'], df_nona['value'], kind='cubic')
    df['cubic_fill'] = f(df['rownum'])
    return df


def knn_mean(ts, n):
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            n_by_2 = np.ceil(n / 2)
            lower = np.max([0, int(i - n_by_2)])
            upper = np.min([len(ts) + 1, int(i + n_by_2)])
            ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
            out[i] = np.nanmean(ts_near)
    return out


def seasonal_mean(ts, n, lr=0.7):
    """
    Compute the mean of corresponding seasonal periods
    ts: 1D array-like of the time series
    n: Seasonal window length of the time series
    """
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            ts_seas = ts[i - 1::-n]  # previous seasons only
            if np.isnan(np.nanmean(ts_seas)):
                ts_seas = np.concatenate([ts[i - 1::-n], ts[i::n]])  # previous and forward
            out[i] = np.nanmean(ts_seas) * lr
    return out


def autocorrelation_fun(df_value, lags=50):
    plot_acf(df_value.tolist(), lags=lags)
    plt.show()
    return acf(df_value, nlags=lags)


def partial_autocorrelation_fun(df_value, lags=50):
    plot_pacf(df_value.tolist(), lags=lags)
    plt.show()
    return pacf(df_value, nlags=lags)


def plot_lags(df_value, n_steps=4):
    fig, axes = plt.subplots(1, n_steps, figsize=(10, 3), dpi=100)
    for i, ax in enumerate(axes.flatten()[:4]):
        lag_plot(df_value, lag=i + 1, ax=ax, c='firebrick')
        ax.set_title('Lag ' + str(i + 1))


def sample_entropy(u, m, r):
    # SampEn(ss.value, m=2, r=0.2 * np.std(ss.value))
    """Compute Sample entropy"""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m_i):
        x = [[u[j] for j in range(i, i + m_i - 1 + 1)] for i in range(N - m_i + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(u)
    return -np.log(_phi(m + 1) / _phi(m))


def moving_average(df_values):
    # Moving Average
    df_ma = df_values.rolling(3, center=True, closed='both').mean()
    return df_ma


def localized_regression(df_values, index_df_values, frac=0.05):
    # Loess Smoothing (5% and 15%)
    df_loess = pd.DataFrame(lowess(df_values, np.arange(len(df_values)), frac=frac)[:, 1],
                            index=index_df_values, columns=['value'])
    return df_loess


def granger_causality(df):
    # It accepts a 2D array with 2 columns as the main argument. The values are in the first column and
    # the predictor (X) is in the second column.
    # The Null hypothesis is: the series in the second column, does not Granger cause the series in the first.
    # If the P-Values are less than a significance level (0.05) then you reject the null hypothesis and conclude that
    # the said lag of X is indeed useful.
    grangercausalitytests(df[['value', 'month']], maxlag=2)


df_data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'],
                      index_col='date')
