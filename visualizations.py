import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

sns.set(style='whitegrid')

LARGE  = (16,7)
MEDIUM = (11,6)
SMALL  = (6,4)

def plot_prices(values, title='Prices', xlabel='Date', ylabel='Price', figsize=MEDIUM):
    """Plot certain prices or returns.

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

def plot_cumulative_returns(returns, figsize=MEDIUM):
    """Plot the cumulative rewards of returns.

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Returns to plot

    figsize : tuple
        Width and height of the plot
    """
    cr = cum_returns(returns)

    plot_prices(cr, title='Cumulative rewards', ylabel='Returns', figsize=figsize)

def plot_drawdown(returns, figsize=MEDIUM):
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
    plt.plot(dd, label='Drawdown')
    plt.plot(mdd, label='Max drawdown')
    plt.title('Drawdown')
    plt.legend()
    plt.plot()

