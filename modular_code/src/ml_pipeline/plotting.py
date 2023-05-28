
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ml_pipeline.outliers import compute_outliers, fill_nulls


def plot_correlation_matrix(df, 
                method = 'pearson', 
                title = None,
                ax = None):

    # create correlation matrix
    correlation_matrix = df.corr(method=method)

    # plot heatmap of correlations
    if ax is None: plt.figure(figsize = (10,8))
    sns.heatmap(correlation_matrix, 
                vmin=-1,
                vmax=1,
                linewidths=1,
                linecolor='k',
                cmap='Purples', 
                annot=True,
                ax=ax)
    if title is not None:
        ax.set_title(title, fontweight='bold')


def plot_univariate_by_target(
                    ts, 
                    target,
                    target_groups = {'Normal':[0], 'Broken or Recovering':[1,2]} 
                    ):

    # split data into two groups based on input
    cuts = {k: target.isin(v) for k,v in target_groups.items()}

    # force a new plot each time
    fig, ax = plt.subplots(figsize=(10,5))

    # plot distributions and add labels
    for k,v in cuts.items():
        sns.kdeplot(ts[v], label=k, ax=ax)
    ax.legend()
    ax.set_title(ts.name, fontweight='bold')


def plot_timeseries_by_target(ts, target):
    # ts is feature series, target is ground truth series

    # create figure with 3 subplots as rows
    fig, axes = plt.subplots(nrows=3, figsize=(12,7), 
                    constrained_layout=True)

    # first row: raw values 
    ts.plot(ax=axes[0])
    axes[0].set_ylabel('Value')
    axes[0].set_title(ts.name, fontweight='bold')

    # second row: first difference of values
    ts.diff().plot(ax=axes[1])
    axes[1].set_ylabel('First difference')

    # third row: target values
    target.plot(ax=axes[2])
    axes[2].set_ylabel(target.name)


def plot_cumsum_of_nulls(ts):

    # plot cumumlative sum of nulls 
    ax = ts.isna().cumsum().plot()
    ax.set_title(f'Null values in {ts.name}', fontweight='bold')
    ax.set_ylabel(f'Cumulative sum')



def plot_subset_of_series(ts, index_loc=None, w=60):

    if index_loc is not None:
        # get integer location of index 
        iloc = ts.index.get_loc(index_loc)

        # get subset of the timeseries
        ts = ts.iloc[iloc-w:iloc+w].copy()

    # make figure and add labels
    ax = ts.plot(figsize=(12,4))
    ax.set_ylabel(ts.name)
    ax.set_title(f'Centered on: {index_loc}', fontweight='bold')

    # return the subset so we can re-use
    return ts






