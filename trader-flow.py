import yfinance as yf
import numpy as np
import pandas as pd

from metaflow import FlowSpec, step, IncludeFile, Parameter
from utils import split_sequence, normalize

class TraderFlow(FlowSpec):
    """
    Metaflow to parse and train the data for the deep q learner
    """

    stock_codes = ['A', 'GOOG', 'INTC', 'TSLA', 'MSFT']

    window_size = Parameter('window',
                    help='The size of the stock windows',
                    default=14)

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

        self.stock_code = self.input

        print(f'Fetching stock {self.stock_code}')

        # Fetch the historic data of the stock
        stock_data = yf.Ticker(self.stock_code)

        stock_history = stock_data.history(
            period='1d', start='2000-1-1', end='2020-2-7')

        # Get the closing prices
        stock_closing = stock_history['Close']

        # Transform into timeseries
        self.stocks = split_sequence(stock_closing, self.window_size)

        # Join all the stock prices
        self.next(self.join_stocks)

    @step
    def join_stocks(self, inputs):
        """
        Join the stock data into a dictionary
        """
        # Create a dictionary for all the stocks
        self.stocks = {input.stock_code: input.stocks for input in inputs}

        # Print all the stocks and their sizes
        for stock_code, stocks in self.stocks.items():
            print(stock_code, stocks.shape)

        self.next(self.end)

    @step
    def end(self):
        """
        Flow is done
        """
        
        print('Done!')

if __name__ == '__main__':
    TraderFlow()
