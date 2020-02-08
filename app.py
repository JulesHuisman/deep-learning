import yfinance as yf
import numpy as np
import pandas as pd

from metaflow import FlowSpec, step, IncludeFile, Parameter

def get_stock(stock_code):
    stock_data = yf.Ticker(stock_code)

    # Get the historical prices
    stock_history = stock_data.history(
        period='1d', start='2010-1-1', end='2020-2-7')

    return stock_history['Close']

class TraderFlow(FlowSpec):
    """
    Metaflow to parse and train the data for the deep q learner
    """

    @step
    def start(self):
        """
        Load the data from the api endpoint
        """

        stocks = ['TSLA', 'MSFT']

        # Get the historical prices
        stock_history = stock_data.history(
            period='1d', start='2010-1-1', end='2020-1-25')

        # Store the stock data
        self.stocks = stock_history

        self.next(self.end)

    @step
    def end(self):
        print(len(self.stocks))

if __name__ == '__main__':
    TraderFlow()
