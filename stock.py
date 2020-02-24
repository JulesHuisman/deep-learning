import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Stock:
    def __init__(self, 
                 ticker, 
                 window_size, 
                 train_size=.3, 
                 normalize=True,
                 diff=True,
                 clip=True,
                 start_date='2010-1-1', 
                 end_date='2020-1-1'):
        
        self.ticker          = ticker
        self.window_size     = window_size
        self.train_size      = train_size
        self.start_date      = start_date
        self.end_date        = end_date
        self.diff            = diff
        self.normalize       = normalize
        self.clip            = clip
        
        self.stock_prices_raw = np.array([])
        self.stock_prices     = np.array([])
        self.stock_prices_1   = np.array([])
        self.stock_prices_n   = np.array([])
        self.stock_seq        = np.array([])
        
        self._fetch_stock()
        self._set_stocks()
        self._sequence()
        
    def _fetch_stock(self):
        # Fetch the historic data of the stock
        stock_data = yf.Ticker(self.ticker)
        
        self.stock_prices_raw = stock_data.history(period='1d', 
                                   start=self.start_date, 
                                   end=self.end_date)
        
        self.stock_prices_raw = self.stock_prices_raw.dropna()['Close'].values
        
    def _set_stocks(self):
        """
        Create different stock array that follow the same index as the sequence data
        """
        self.stock_prices = np.array(self.stock_prices_raw[(self.window_size-1):])
        self.stock_prices_1 = np.array(self.stock_prices_raw[(self.window_size-2):])
    
    def _normalize(self, stocks):
        # The size of the train split
        train_split = int(len(stocks)*self.train_size)
        train_set = stocks[:train_split]
        
        # Standardscale the data
        scaler = StandardScaler()
        scaler.fit(train_set.reshape(-1,1))
        
        return scaler.transform(stocks.reshape(-1,1)).flatten()

    @staticmethod
    def _diff(stocks):
        return np.diff(stocks, 1, prepend=[stocks[0]])

    @staticmethod
    def _clip(stocks):
        mean = np.mean(stocks)
        std = np.std(stocks)
        
        return np.tanh(stocks)
        # return np.clip(stocks, mean-std, mean+std)

    @staticmethod
    def _split_sequence(sequence, n_steps):
        """
        Create timeseries from array
        """

        X = []
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence):
                break
            seq_x = sequence[i:end_ix]
            X.append(seq_x)

        return np.array(X)
        
    def _sequence(self):
        stocks = self.stock_prices_raw
        
        if self.diff:
            stocks = self._diff(stocks)
        
        if self.normalize:
            stocks = self._normalize(stocks)

        if self.clip:
            stocks = self._clip(stocks)
        
        self.stock_seq = self._split_sequence(stocks, self.window_size)