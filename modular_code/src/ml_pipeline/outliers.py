import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt



def compute_bounds_std(ts, std_threshold=3):

    # take mean and standard deviation of time series
    mean = ts.mean()
    std = ts.std()

    # use this to compute upper and lower bounds based on our threshold
    lower_bound = mean - (std_threshold * std)
    upper_bound = mean + (std_threshold * std)
    return lower_bound, upper_bound


def compute_bounds_quantiles(ts,  bounds = [0.25, 0.75], range_multiplier=1.5):

    # get the quartiles and interquartile range
    lower_quantile = ts.quantile(bounds[0])
    upper_quantile = ts.quantile(bounds[1])
    quantile_range = upper_quantile - lower_quantile

    # compute bounds based on distnace from qunatiles
    upper_bound = upper_quantile + (range_multiplier * quantile_range)
    lower_bound = lower_quantile - (range_multiplier * quantile_range)
    return lower_bound, upper_bound


def compute_outliers(ts, 
                    bounds_function=compute_bounds_std, 
                    kwargs={'std_threshold':3},    
                    n_passes = 2               
                    ):
    # copy input time series
    ts_fix = ts.copy()

    # for the number of passes we want
    for _ in range(n_passes):

        # rolling window 
        lower_bound, upper_bound = bounds_function(ts_fix, **kwargs)

        # get boolean mask for any point outside fo bounds
        outliers = ((ts_fix < lower_bound) | (ts_fix > upper_bound))
        
        # replace outliers with nulls, but only if they are in the middle
        ts_fix = null_outliers(ts_fix, outliers, only_inner=True)

    return ts_fix, outliers, lower_bound, upper_bound



def null_outliers(ts, outliers, only_inner=True):

    # convert outliers (True/False) into nulls or 1s
    outliers_as_nulls = outliers.apply(lambda x: np.nan if x else 1)

    if only_inner:
        # interpolate this bool so that only nulls at the very start/end will remain
        # then take bool maask of this which will be false if outlier occured at very start/end
        valid_outliers = (outliers_as_nulls.interpolate(limit_area='inside') == 1)

        # apply bool mask to reduce outliers to only inner outlier
        outliers = (outliers * valid_outliers)

    # copy input series
    ts_fixed = ts.copy()

    # replace outliers will nulls
    ts_fixed.loc[outliers.values] = np.nan

    return ts_fixed



prior_for_missing = {'Temperature': 0.0,
                    'Humidity': 0.0,
                    'RPM': 0.0,
                    'Vibrations': 0.0,
                    'Pressure': 0.0,
                    'Sensor1': 0.0,
                    'Sensor2': 0.0,
                    'Sensor3': 0.0,
                    'Sensor4': 0.0,
                    'Sensor5': 0.0,
                    'Sensor6': 0.0}


def fill_outer_nulls(ts):
    # backfill will replace nulls at very start with first non-null
    ts = ts.interpolate(method='bfill')
    # fowardfill will replace nulls at very end with last non-null
    ts = ts.interpolate(method='ffill')
    return ts

def fill_inner_nulls(ts):
    # middle nulls will be linearly interpolated between neighboring non-nulls
    ts = ts.interpolate(method='linear', limit_area='inside')
    return ts

def fill_with_priors(ts, defaults=prior_for_missing):
    # get default value for this series
    prior_val = defaults.get(ts.name, 0)
    # fill missing values
    ts = ts.replace(np.nan, prior_val)
    return ts


def fill_nulls(ts):
    # this is necessary to avoid a known bug in pandas when interpolating on 
    # a series with datetimes
    ts_fixed = pd.Series(data=ts.values.astype(float))
    ts_fixed.name = ts.name

    # fix nulls by calling other functions
    ts_fixed = fill_inner_nulls(ts_fixed)
    ts_fixed = fill_outer_nulls(ts_fixed)
    ts_fixed = fill_with_priors(ts_fixed)

    # put original index back in place
    ts_fixed.index = ts.index

    return ts_fixed


def clean_outliers_and_fill_missing(df):
    # function that applies our outlier detection and
    #  misisng imputation to all columns in a dataframe
    for name, col in df.iteritems():
        col, *_ = compute_outliers(col)
        col = fill_nulls(col)
        df[name] = col
    return df


def plot_outliers(ts, bounds_function, kwargs={}):

    # compute outlier
    _, outliers, lower_bound, upper_bound = compute_outliers(ts, bounds_function, kwargs, n_passes=1)

    # plot main series 
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(ts, color='k', alpha=0.5, label='timeseries')

    # plot bounds (they are the same color and style so we only need to label one)
    ax.axhline(lower_bound, color='b', alpha=0.75, linestyle='--', label='bounds')
    ax.axhline(upper_bound, color='b', alpha=0.75, linestyle='--')

    # add outliers as dots
    ax.plot(ts[outliers], marker='o', color='r', linewidth=0, label='outliers')

    # add legend
    ax.legend()


def plot_with_nulls_filled(ts):

    # compute filled values for this series
    ts_filled = fill_nulls(ts)

    # get just the part we filled
    filled_only = ts_filled[ts.isna()]

    # plot the original series and filled part 
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(ts, color='k', linestyle='--',  marker='o',
                label='series with missing')
    ax.plot(filled_only, color='r', marker='o',
                linestyle='-',  label='filled values')
    ax.legend()
    ax.set_ylabel(ts.name)
    
    return ts_filled


