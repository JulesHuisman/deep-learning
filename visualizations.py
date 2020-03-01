import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

sns.set(style='whitegrid')

LARGE  = (18,7)
MEDIUM = (11,6)
SMALL  = (6,4)

def plot(values, title='', xlabel='Date', ylabel='', figsize=LARGE):
    """Simple line plot.

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Returns to plot

    title : str
        Title of the plot

    xlabel : str
        Label on the x axis

    ylabel : str
        Label on the y axis

    figsize : tuple
        Width and height of the plot
    """
    ax = values.plot(title=title, figsize=figsize)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.legend(loc='upper left')

def plot_cumulative_returns(returns, figsize=LARGE):
    """Plot the cumulative rewards of returns.

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Returns to plot

    title : str
        Title of the plot

    figsize : tuple
        Width and height of the plot
    """
    cr = cum_returns(returns)

    plot(cr, title='Cumulative rewards', ylabel='Returns', figsize=figsize)

    plt.legend(loc='upper left')

def plot_drawdown(returns, title='Drawdown', figsize=LARGE):
    """Plot the cumulative rewards and drawdown of a return.

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Returns to plot

    figsize : tuple
        Width and height of the plot
    """
    cr = cum_returns(returns)
    dd = drawdown(returns)
    mdd = max_drawdown(returns)

    plt.figure(figsize=figsize)

    plt.plot(cr, label='Cumulative simple returns')
    plt.plot(dd, 'r-', label='Drawdown')
    plt.plot(mdd, 'r-.', label='Max drawdown')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.plot()

def show_sharpe_ratio(returns):
    """Show a tabel with the sharpe ratios of multiple assets.

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Returns to calculate sharpe ratios for

    Returns
    -------
    ratios : pd.DataFrame
        Dataframe with sharpe ratios.
    """
    sr = sharpe_ratio(returns)

    df = pd.DataFrame(sr, columns=['Sharpe ratio'])

    cm = sns.light_palette("green", as_cmap=True)

    return df.style.background_gradient(cmap=cm)

def plot_positions(positions):
    values = [smooth(positions[ticker], 50) for ticker in positions.columns]

    # Make the plot
    plt.figure(figsize=LARGE)
    plt.stackplot(range(0,len(positions)), *values, labels=positions.columns)
    plt.legend(reversed(plt.legend().legendHandles), reversed(positions.columns), loc='upper left')
    plt.margins(0,0)
    plt.title('Position of all assets')
    plt.xlabel('Iteration')
    plt.ylabel('Position percentage')
    plt.show()
    

