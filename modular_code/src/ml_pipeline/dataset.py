import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import SlidingWindowSplitter
from ml_pipeline.outliers import clean_outliers_and_fill_missing


def load_excel(path, select_sheet=None):

    # read the excel file
    xlsx = pd.ExcelFile(path)
    
    # dictionary with one key/value for each sheet
    xl_dict = {sheet: xlsx.parse(sheet) for sheet in xlsx.sheet_names}
    
    # return either entire dict, or just one sheet  
    if select_sheet is None:
        return xl_dict
    else: 
        return xl_dict[select_sheet]


def time_train_test_split(df, proportion = 0.5):

    # cut point is based on proportion and length
    cut = int(len(df) * proportion)

    # make cut
    train = df.iloc[:cut]
    test = df.iloc[cut:]

    return train, test


def create_windows(df, 
                target_column='Status', 
                window_kwargs = {'window_length':30, 'step_length':1, 'fh':0},
                ):

    # separate the ground truth from all other columns
    y_col = df[target_column]
    X_col = df.drop(target_column, axis=1)

    # create splitter using kwargs
    splitter = SlidingWindowSplitter(**window_kwargs)

    # get indices
    split = splitter.split(df)

    # create arrays of correspondning X and y
    X, y = [], []

    # fill X and y using each split
    for X_idx, y_idx in split:
        X.append(X_col.iloc[X_idx])
        y.append(y_col.iloc[y_idx])

    return X, y


def add_time_features(df, 
                add_features= ['month', 'day_of_week', 'hour', 'hours_since'], 
                center_hour=12):
    # df must be a dataframe with a datetime index

    def get_abs_delta(x):
        hr = x.hour
        fwd = np.abs(hr - center_hour)
        bck = np.abs((24+hr) - center_hour)
        return min(fwd, bck)

    time_funcs = {
        # this adds month of year (January=1, ... , December=12)
        'month': lambda x: x.month,

        # this add day of week (Monday=0, ... , Sunday=6)
        'day_of_week': lambda x: x.day_of_week,

        # add hour of day (0-23)
        'hour': lambda x: x.hour,

        # we can also 'smooth' hour by instead having absolute hours from 
        # the centered hour, by default we use midday (12)
        'hours_since': get_abs_delta}

    # add chose features 
    for k in add_features:
        df[k] = df.index.map(time_funcs[k])

    return df


def add_first_differences(df, target='Status', excl='_isnull'):

    # get list of columns to exclude
    excl_cols = [c for c in df.columns if (excl in c)|(target in c)]

    # take first difference of remaining columns
    df1 = df[df.columns.difference(excl_cols)]
    df2 = df1.diff().replace(np.nan,0) # replace first row with zero
    df2.columns = [c + '_diff' for c in df2.columns]

    return df1.join(df2).join(df[excl_cols])

def drop_columns(df, to_drop = ['Sensor1',]):
    # drop any columns in the list that are in the dataframe
    return df.drop(set(to_drop).intersection(df.columns), axis=1)


def preprocess(df, add_first_diff=True):

    # drop the redundant column
    df = drop_columns(df, to_drop = ['Sensor1'])

    # deal with outliers and missing values
    df = clean_outliers_and_fill_missing(df)

    if add_first_diff:
        # add first difference columns
        df = add_first_differences(df)

    return df
