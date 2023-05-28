import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt
import seaborn as sns


def generate_sinewave(
                    frequency = 2, # cycles per hour
                    amplitude = 1, 
                    sample_rate = 60, # we are doing minutes per hour
                    n_time = 60, # number of time steps to generate
                    ):

    # one sinewave is half a cycle so we multiple by 2
    div = (frequency*2)

    # this is the rate of change of our sinewave
    step_size = np.pi/(sample_rate/ div )

    # make signal
    sinewave = np.sin(np.arange(0, n_time*step_size, step=step_size))

    # adjust amplitude
    return sinewave * amplitude




def plot_waveform(signal, 
                x_label='Minutes', 
                y_label='Amplitude',
                xstep=15, ax = None, add_zero=True):

    # length of signal
    n = len(signal)

    # plot a signal 
    if ax is None:
        plt.figure(figsize=(12,3))
        with sns.axes_style("whitegrid"):
            ax = sns.lineplot(x=range(n), y=signal)
    else:
        ax = sns.lineplot(x=range(n), y=signal, ax=ax)

    # label axes
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    # place xticks every xstpes
    ax.set_xticks(range(0, n+1, xstep))

    # add horizontal line at zero
    if add_zero:
        ax.axhline(0, linestyle='--', color='k')

    return ax


def decompose_series(ts, sample_rate=60, absolute=False):

    # we must give the fft functions a numpy array
    ts_vals = ts.values if isinstance(ts, pd.Series) else ts

    # this simple function gives us the list of frequencies, given our length and sample rate
    xf = rfftfreq(len(ts_vals), 1 / sample_rate)

    # get power of each for frequency band
    yf = rfft(ts_vals) # n/2 entries 

    # optional: convert from complex to real numbers
    if absolute:
        yf = np.abs(yf)

    return pd.Series(index=xf, data=yf, name='decomposition')


def plot_signal_frequency_domain(fd,  ax=None):

    # plot frequency domain
    if ax is None:
        fig,ax = plt.subplots(figsize=(12,3))
    ax.bar(x=fd.index, height=fd.values)

    # add labels
    ax.set_ylabel('Power')
    ax.set_xlabel('Frequency')



def filter_frequencies(ts, min_freq = None, max_freq = None):

    # decompose the time series (keeping complex numbres, hence aboslute=False)
    fd = decompose_series(ts, absolute=False)

    # all frequency bands above max_freq we set to zero
    fd[fd.index > max_freq] = 0

    # all frequency bands below min_freq we set to zero
    fd[fd.index < min_freq] = 0

    # this function recontructs a time series 
    vals = irfft(fd.values)

    # format as series 
    ts_clean = pd.Series(data=vals)

    # if original input was a series, use the same index
    if isinstance(ts, pd.Series):
        ts_clean.index = ts.index
        ts_clean.name = ts.name

    return ts_clean