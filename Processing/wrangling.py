import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn import preprocessing
from statsmodels.nonparametric.smoothers_lowess import lowess


def write_dataframe(df, filename, encoding=None, sheet_name=""):
    extension = filename.split(".")[1]
    if extension == "csv":
        df.to_csv(filename, index=False, encoding=encoding, sep=';')
    else:
        df.to_excel(filename, sheet_name=sheet_name, index=False)
    return


def read_dataframe(filename, sheetname=None, columns=None, index_col=None, encoding=None, sep=';'):
    extension = filename.split(".")[1]
    if extension == "csv":
        # Read dataframe
        df = pd.read_csv(filename, index_col=index_col, encoding=encoding, sep=sep)
        if columns is None:
            return df
        else:
            return df[columns]
    else:
        # Read dataframe
        df = pd.read_excel(filename, sheet_name=sheetname, index_col=index_col, engine='openpyxl')
        if columns is None:
            return df
        else:
            return df[columns]

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


def moving_average(df_values):
    # Moving Average
    df_ma = df_values.rolling(3, center=True, closed='both').mean()
    return df_ma


def localized_regression(df_values, index_df_values, frac=0.05):
    # Loess Smoothing (5% and 15%)
    df_loess = pd.DataFrame(lowess(df_values, np.arange(len(df_values)), frac=frac)[:, 1],
                            index=index_df_values, columns=['value'])
    return df_loess


def cubic_interpollation_fill(df):
    df['rownum'] = np.arange(df.shape[0])
    df_nona = df.dropna(subset=['value'])
    f = interp1d(df_nona['rownum'], df_nona['value'], kind='cubic')
    df['cubic_fill'] = f(df['rownum'])
    return df


def remove_duplicates(df, dups_columns):
    df.drop_duplicates(dups_columns, inplace=True)
    return df


def remove_na(df, nan_columns):
    df.dropna(subset=nan_columns, axis=0, inplace=True)
    return df


def zero_fill_na(df, zero_fill_columns):
    for i in zero_fill_columns:
        df[i] = df[i].fillna(0)
    return df


def custom_fill_na(df, custom_fill_column, custom_fill_char):
    for col, char in zip(custom_fill_column, custom_fill_char):
        df[col] = df[col].fillna(char)
    return df


def convert_dash_to_na(df):
    df.replace('-', np.nan, inplace=True)
    return df


def convert_char_to_numeric(df):
    df = df.apply(pd.to_numeric, errors='ignore')
    return df


def convert_char_na_to_na(df):
    df.replace('nan', np.nan, inplace=True)
    return df
    

def convert_discrete_to_categorical(df, categorical_cols):
    le = preprocessing.LabelEncoder()
    for col in categorical_cols:
        df[col+'_CATEGORICAL'] = le.fit_transform(df[col].astype(str))
    return df


def convert_continuous_to_categorical(df, categorical_cols, bins):
    for col in categorical_cols:
        df[col+'_CATEGORICAL'] = pd.cut(df[col], bins=bins, labels=[i for i in range(len(bins))]).astype(int)
    return df


def outliers_score(ys, way='lof'):
    if way == 'z-score':
        threshold_s = 3
        mean_y = ys.mean()
        stdev_y = ys.std()
        z_scores = (ys - mean_y) / stdev_y
        outlier_index = np.where(np.abs(z_scores) > threshold_s)[0]
    elif way == 'isolation-forest':
        from sklearn.ensemble import IsolationForest
        ifor = IsolationForest(contamination=.1, random_state=1)
        result = ifor.fit_predict(np.array(ys).reshape(-1, 1))
        outlier_index = np.where(result == -1)
    else:
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=35, contamination=.1)
        result = lof.fit_predict(np.array(ys).reshape(-1, 1))
        outlier_index = np.where(result == -1)
    return outlier_index


def remove_outliers(y):
    outlier_index = outliers_score(y, way='z-score')
    y.drop(outlier_index, inplace=True)
    return y


def standardize_dataframe(df, columns):
    # Standardization
    x = df.values  # returns a numpy array
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=columns)
    return df, scaler


def unstandardize_dataframe(scaler, df, columns, label_str='Label'):
    label_series = df[label_str]
    x = df.iloc[:, :-1].values  # returns a numpy array
    x_scaled = scaler.inverse_transform(x)
    df = pd.DataFrame(x_scaled, columns=columns)
    df[label_str] = label_series
    return df


def normalize_dataframe(df, columns):
    # Normalization
    x = df.values  # returns a numpy array
    scaler = preprocessing.Normalizer()
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=columns)
    return df, scaler


def unnormalize_dataframe(normalizer, df, columns, label_str='Label'):
    label_series = df[label_str]
    x = df.iloc[:, :-1].values  # returns a numpy array
    x_scaled = normalizer.inverse_transform(x)
    df = pd.DataFrame(x_scaled, columns=columns)
    df[label_str] = label_series
    return df
