import pandas as pd
import numpy as np
import warnings

eps = np.finfo(float).eps

def convert(values):
    """Convert numpy arrays or lists to pandas series.

    Parameters
    ----------
    values : pd.DataFrame | pd.Series | np.ndarray
        Array to transform

    Returns
    -------
    series : pd.Series
        Pandas series.
    """
    if isinstance(values, (list,np.ndarray)):
        return pd.Series(values)
    else:
        return values

def simple_returns(values):
    """Converts values to simple returns.

    Parameters
    ----------
    values : pd.DataFrame | pd.Series | np.ndarray
        Values you want to convert to simple returns

    Returns
    -------
    simple_return : pd.Series | pd.DataFrame
        Simple returns.
    """
    values = convert(values)
    
    return ((values / values.shift(1)) - 1).fillna(0)

def log_returns(values):
    """Converts values to log returns.

    Parameters
    ----------
    values : pd.DataFrame | pd.Series | np.ndarray
        Values you want to convert to log returns

    Returns
    -------
    log_returns : pd.Series | pd.DataFrame
        Log returns.
    """
    values = convert(values)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.log(values / values.shift(1)).fillna(0)

def cum_returns(returns):
    """Computes cumulative returns from simple returns.

    Parameters
    ----------
    returns : np.ndarray | pd.Series | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    cumulative_returns : np.ndarray | pd.Series | pd.DataFrame
        Cumulative returns.

    # https://github.com/filangel/qtrader
    """
    out = returns.copy()
    out = np.add(out, 1)
    out = out.cumprod(axis=0)
    out = np.subtract(out, 1)

    return out

def sharpe_ratio(returns):
    """Computes Sharpe Ratio from simple returns.

    Parameters
    ----------
    returns : np.ndarray | pd.Series | pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    sharpe_ratio : float | np.ndarray | pd.Series
        Sharpe ratio.

    # https://github.com/filangel/qtrader
    """
    return np.sqrt(len(returns)) * np.mean(returns, axis=0) / (np.std(returns, axis=0) + eps)

def drawdown(returns):
    """Computes Drawdown given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    drawdown : pandas.Series
        Drawdown of strategy.

    # https://github.com/filangel/qtrader
    """
    _cum_returns = cum_returns(returns)
    expanding_max = _cum_returns.expanding(1).max()
    drawdown = expanding_max - _cum_returns
    drawdown *= -1

    return drawdown

def max_drawdown(returns):
    """Computes Max Drawdown given simple returns.

    Parameters
    ----------
    returns : pandas.Series
        Returns of the strategy as a percentage, noncumulative.

    Returns
    -------
    max_drawdown : pandas.Series
        Max drawdown of strategy.

    # https://github.com/filangel/qtrader
    """
    _drawdown = drawdown(returns)
    return _drawdown.expanding(1).min()