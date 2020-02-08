import yfinance as yf
import numpy as np
import pandas as pd

from metaflow import FlowSpec, step, IncludeFile, Parameter

# def get_stock(stock_code):
#     stock_data = yf.Ticker(stock_code)

#     # Get the historical prices
#     stock_history = stock_data.history(
#         period='1d', start='2010-1-1', end='2020-2-7')

#     return stock_history['Close']

class TraderFlow(FlowSpec):
    """
    Metaflow to parse and train the data for the deep q learner
    """

    stock_codes = ['A', 'GOOG', 'INTC', 'TSLA', 'MSFT']

    @step
    def start(self):
        """
        Defines the stocks to fetch
        """

        # Fan out the fetching of stocks
        self.next(self.get_stock, foreach='stock_codes')

    @step 
    def get_stock(self):
        """
        Fetch the historic data of a stock and return the closing prices
        """

        stock_code = self.input

        print(f'Fetching stock {stock_code}')

        # Fetch the historic data of the stock
        stock_data = yf.Ticker(stock_code)

        stock_history = stock_data.history(
            period='1d', start='2000-1-1', end='2020-2-7')

        # Get the closing prices
        self.stocks = stock_history['Close']

        # Join all the stock prices
        self.next(self.join_stocks)

    @step
    def join_stocks(self, inputs):
        self.df = pd.concat([input.stocks for input in inputs], axis=1)
        self.df.columns = self.stock_codes

        self.next(self.end)

    @step
    def end(self):
        print(f'Final shape: {self.df.shape}')

if __name__ == '__main__':
    TraderFlow()
