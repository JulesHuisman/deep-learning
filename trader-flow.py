import yfinance as yf
import numpy as np
import pandas as pd
import pickle

from metaflow import FlowSpec, step, IncludeFile, Parameter
from utils import split_sequence
from sklearn.preprocessing import StandardScaler

class TraderFlow(FlowSpec):
    """
    Metaflow to parse and train the data for the deep q learner
    """

    stock_codes = ['^GSPC', 'AAPL']

    window_size = Parameter('window',
                    help='The size of the stock windows',
                    default=200)

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

        train = stock_data.history(
            period='1d', start='2001-1-1', end='2005-1-1')

        test = stock_data.history(
            period='1d', start='2005-1-1', end='2015-12-31')

        # Get the price difference
        train_diff = train['Close'].diff()[1:].values
        test_diff = test['Close'].diff()[1:].values

        # Normalize
        scaler = StandardScaler()
        train_diff = scaler.fit_transform(train_diff.reshape(-1,1)).flatten()
        test_diff = scaler.transform(test_diff.reshape(-1,1)).flatten()

        # Raw prices
        train_price = train['Close'].values
        test_price = test['Close'].values

        # Get timeseries
        self.train_seq = split_sequence(train_diff, self.window_size)
        self.test_seq  = split_sequence(test_diff, self.window_size)

        self.train_price = train_price
        self.test_price = test_price

        # Join all the stock prices
        self.next(self.join_stocks)

    @step
    def join_stocks(self, inputs):
        """
        Join the stock data into a dictionary
        """
        # Create a dictionary for all the stocks
        self.stocks = {input.stock_code: {
            'train': {
                'seq': input.train_seq,
                'price': input.train_price
            },
            'test': {
                'seq': input.test_seq,
                'price': input.test_price
            }} for input in inputs}

        # Print all the stocks and their sizes
        for stock_code, stocks in self.stocks.items():
            print(stock_code, 'train', stocks['train']['seq'].shape, 'test', stocks['test']['seq'].shape)

        self.next(self.end)

    @step
    def end(self):
        """
        Store the data
        """
        with open('data/data.pkl', 'wb') as handle:
            pickle.dump(self.stocks, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Done!')

if __name__ == '__main__':
    TraderFlow()
