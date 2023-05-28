import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ml_pipeline.outliers import compute_outliers, fill_nulls

# maps numbers in Status column to names of status
status_name = {0:'Normal', 1:'Broke', 2:'Recovering'}

def check_target_split(train, test, target):
    # target must be the name of a column in both train and test
    train_count = train[target].value_counts().rename('train')
    test_count = test[target].value_counts().rename('test')
    comparison = pd.concat([train_count, test_count],axis=1).sort_index()
    comparison.index.name = target
    print(comparison.rename(index=status_name))


# maps numbers returned by pd.DatetimeIndex.day_of_week to day names
day_of_week_name = {n:x for n, x in enumerate(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])}
    
def check_number_of_full_days(datetime_series):

    # count of entries for each day
    count = datetime_series.day_of_week.value_counts()

    # count of days, assuming series is in minutes
    count /= (60*24)
    
    # sort by day and rename     
    print(count.sort_index().rename(day_of_week_name))



