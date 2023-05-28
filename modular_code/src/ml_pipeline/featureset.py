import numpy as np
import pandas as pd

def get_slope(ts):
    # return slope (coefficient of linear trendline) of a series
    xs = range(len(ts))
    return np.polyfit(xs, ts, 1)[0]


def summary_stats(df):
    # function to append suffix to column names
    mapcol = lambda x: {c:c+'_'+x for c in df.columns}

    # mean of each series
    means = df.mean().rename(mapcol('mean'))

    # standard deviation of each series
    stds = df.std().rename(mapcol('std'))

    # slope of each series
    slopes = df.apply(get_slope, axis=0).rename(mapcol('slope'))

    # concatentate into single feature vector 
    return pd.concat([means,stds,slopes])


def make_summary_stats_input(X, y):

    # concatenate ys into one series
    y = pd.concat(y)

    # get summary stat for each X entry
    X = [summary_stats(x) for x in X]

    # concatenate and set index to y
    X = pd.concat(X, axis=1).T
    X.index = y.index

    return X, y


def make_series_input(X, y):

    # concatenate ys into one series
    y = pd.concat(y)

    # convert to list 
    X = [pd.Series({nam:ts.reset_index(drop=True) for nam, ts in df.iteritems()}) for df in X]

    # concatenate and set index to y
    X = pd.concat(X,axis=1).T
    X.index = y.index

    return X, y