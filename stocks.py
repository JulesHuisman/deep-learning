import numpy as np
import pandas as pd
import yfinance as yf
import os

from utils import *

class Stocks:
    def __init__(self, tickers, start='2005-1-1', end='2018-4-1'):
        self.tickers = tickers

        stocks = self._fetch_stocks(tickers)
        stocks = self._merge_stocks(stocks)
        stocks = self._select_timeframe(stocks, start, end)

        self._stocks = stocks
        
    @property
    def stocks(self):
        """
        Returns the closing stock prices
        """
        return self._stocks

    @property
    def simple_returns(self):
        """
        Returns the simple returns of the stocks
        """
        return simple_returns(self._stocks)

    @property
    def log_returns(self):
        """
        Returns the log returns of the stocks
        """
        return log_returns(self._stocks)

    def _fetch_stocks(self, tickers):
        """
        Either load stocks from storage or fetch from yahoo finance
        """
        stocks = []
        
        # Loop through all the different stocks
        for ticker in tickers:
            # If the stock does not exists on storage
            if not self._stock_exists(ticker):
                # Fetch it from yahoo finance
                stock = self._fetch_stock(ticker)
                self._store_stock(stock, ticker)
            # If it does exist in storage, load it
            else:
                stock = self._load_stock(ticker)
                
            # Rename the closing price to the ticker symbol
            stock = stock['Close'].rename(ticker)
                
            stocks.append(stock)
            
        return stocks
    
    @staticmethod
    def _merge_stocks(stocks):
        """
        Merges all the stock prices into one dataframe
        """
        return pd.concat(stocks, axis=1)
    
    @staticmethod
    def _select_timeframe(stocks, start, end):
        """
        Merges all the stock prices into one dataframe
        """
        return stocks.loc[start:end]
                
    @staticmethod
    def _load_stock(ticker):
        """
        Load the stock from storage
        """
        stock = pd.read_pickle(f'data/stocks/{ticker}.pkl')
        return stock
                
    @staticmethod
    def _fetch_stock(ticker):
        """
        Fetch the stock data from yahoo finance
        """
        stock = yf.Ticker(ticker)
        history = stock.history(period="max")
        
        return history
    
    @staticmethod
    def _store_stock(stock, ticker):
        """
        Store the stock on storage
        """
        filename = f'data/stocks/{ticker}.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        stock.to_pickle(f'data/stocks/{ticker}.pkl')
            
    @staticmethod
    def _stock_exists(ticker):
        """
        Check if the file exists on storage
        """
        return os.path.isfile(f'data/stocks/{ticker}.pkl')