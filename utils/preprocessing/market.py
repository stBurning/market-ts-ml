import logging
import numpy as np
import pandas as pd
from typing import Optional, Union
from collections.abc import Iterable
from sklearn.base import BaseEstimator, TransformerMixin


class OHLCVDataProcessor(BaseEstimator, TransformerMixin):
    """
    Class for preprocessing time-series financial OHCLV data
    """

    def __init__(self,
                 volatile_periods=None,
                 ma_periods=None,
                 volumes=None) -> None:
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())

        if volumes is None:
            volumes = ['Volume']

        if ma_periods is None:
            ma_periods = [5, 10, 20, 30, 60]
        if volatile_periods is None:
            volatile_periods = [5, 10, 20, 30, 60]

        self.volumes = volumes
        self.volatile_periods = volatile_periods
        self.ma_periods = ma_periods
        self.data = None

    def get_feature_names_out(self, *args, **params):
        return self.data.columns
        

    @classmethod
    def build_moving_average(cls, series: pd.Series, period=30, ema=False, alpha=0.1):
        if ema:
            return series.ewm(alpha=alpha, adjust=False).mean()
        else:
            return series.rolling(period).mean()

    @classmethod
    def build_candle_shape_features(cls, df):
        df = df.copy(deep=True)
        df['BodyShapeRatio'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
        df['HighOpenShapeRatio'] = (df['High'] - df['Open']) / (df['High'] - df['Low'])
        df['HighCloseShapeRatio'] = (df['High'] - df['Close']) / (df['High'] - df['Low'])
        df['LowOpenShapeRatio'] = (df['Open'] - df['Low']) / (df['High'] - df['Low'])
        df['LowCloseShapeRatio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['HighWickRatio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'])
        df['LowWickRatio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'])
        return df

    @classmethod
    def build_rsi(cls, series: pd.Series, periods: int = 14, ema: bool = True):
        close_delta = series.diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        if ema:
            ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
            ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        else:
            ma_up = up.rolling(window=periods, adjust=False).mean()
            ma_down = down.rolling(window=periods, adjust=False).mean()
        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))
        return rsi

    @classmethod
    def build_stochastic_oscillator(cls, df, k_period, d_period):
        df = df.copy()
        df['n_high'] = df['High'].rolling(k_period).max()
        df['n_low'] = df['Low'].rolling(k_period).min()
        df['%K'] = (df['Close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])
        df['%D'] = df['%K'].rolling(d_period).mean()
        return df['%K'], df['%D']

    @classmethod
    def build_volatile(cls, returns: pd.Series, period=60):
        return returns.rolling(period).std() * np.sqrt(period)

    @classmethod
    def build_bollinger_bands(cls, series: pd.Series, period=20):
        """
        Вычисление полос Бойлленджера
        :param series:
        :param period:
        :return:
        """
        ma_series = series.rolling(window=period).mean()
        std_series = series.rolling(window=period).std()
        upper_band = ma_series + (std_series * 2)
        lower_band = ma_series - (std_series * 2)
        location = (series - ma_series) / (upper_band - lower_band)
        return upper_band, lower_band, location

    @classmethod
    def build_var(cls, series: pd.Series, period: int = 250, low_q: float = 0.01, high_q: float = 0.99):
        """
        Вычисление Value-at-Risk по цене или по доходности

        :param series: цена или доходность инструмента
        :param period: размер окна
        :param high_q: верхний квантиль
        :param low_q: нижний квантиль
        :return:
        """
        var_low = series.rolling(period).apply(lambda x: np.quantile(x, q=low_q))
        var_high = series.rolling(period).apply(lambda x: np.quantile(x, q=high_q))
        relative = (series - var_low) / (var_high - var_low)
        return var_low, var_high, relative

    @classmethod
    def build_rolling_mean_relative(cls, series: pd.Series, window: int = 5):
        return series / series.rolling(window, closed='left').mean()

    @classmethod
    def build_returns(cls, df: Union[pd.DataFrame, pd.Series], index: Optional[str] = None, target="Close",
                      cumulative_periods=None):

        if cumulative_periods is None:
            cumulative_periods = [1, 5, 10, 30, 60, 90]
        df = df.copy()

        if isinstance(df, pd.Series):
            target = df.name
            df = df.to_frame()

        if index is None:
            df = df.sort_index(ascending=True)
        else:
            df.sort_values(by=index, ascending=True)

        df["Return"] = df[target].pct_change()
        df["LogReturn"] = np.log(1 + df["Return"])
        df["Return T+1"] = df[target].pct_change().shift(-1)
        df['LogReturn T+1'] = np.log(1 + df['Return T+1'])

        for i in cumulative_periods:
            df[f'CR [{i}]'] = ((1 + df['Return']).rolling(i).apply(np.prod) - 1).shift(-i)
        return df

    @classmethod
    def __check_data_structure(cls, df):
        assert isinstance(df, pd.DataFrame), "An argument has to be an instance of pandas DataFrame."

        columns = df.columns

        assert ("Close" in columns
                and "Open" in columns
                and "High" in columns
                and "Low" in columns
                and "Volume" in columns), "Expected Open, Close, High, Low and Volume columns."

    def fit(self, X, y=None):
        return self
           

    def transform(self, X, y=None):
        self.__check_data_structure(X)
        self.data = X.copy()

        # Скользящие средние
        for period in self.ma_periods:
            self.data[f'MA [{period}]'] = self.build_moving_average(self.data['Close'], period=period)
        self.data[f'EMA'] = self.build_moving_average(self.data['Close'], ema=True)
        for period in self.ma_periods:
            self.data[f'Close-MA [{period}]'] = self.data['Close'] - self.data[f'MA [{period}]']
        self.data[f'Close-EMA'] = self.data['Close'] - self.data['EMA']
        for period in self.ma_periods:
            self.data[f'd(Close-MA [{period}])'] = self.data[f'Close-MA [{period}]'] / self.data['Close']
        self.data[f'd(Close-EMA)'] = self.data[f'Close-EMA'] / self.data['Close']

        # RSI
        self.data['RSI'] = self.build_rsi(self.data['Close'], periods=14, ema=True)
        self.data['dRSI'] = self.data['RSI'].pct_change()
        self.data['VRSI'] = self.build_rsi(self.data['Volume'], periods=14, ema=True)
        self.data['dVRSI'] = self.data['VRSI'].pct_change()

        # Стохастический осциллятор
        self.data['%K'], self.data['%D'] = self.build_stochastic_oscillator(self.data, 26, 12)
        self.data['%K-%D'] = self.data['%K'] - self.data['%D']

        # Объемные характеристики
        for volume in self.volumes:
            self.data[f'{volume}Change'] = self.data[volume].pct_change()
            for window in [6, 12, 24, 48]:
                self.data[f'{volume}MeanRatio [{window}]'] = self.build_rolling_mean_relative(self.data[volume], window)

        # Свечные характеристики
        self.data = self.build_candle_shape_features(self.data)

        # Доходность
        self.data = self.build_returns(self.data, index=None, target="Close")

        # Волатильность и стандартное отклонение
        self.data['Volatility'] = self.build_volatile(self.data['Return'])
        for period in self.volatile_periods:
            self.data[f'STD [{period}]'] = self.data['Close'].rolling(period).std()

        # Value-at-Risk
        for risk in [1, 5]:
            (self.data[f'VaR.Low [Return, 250, {risk}%]'],
             self.data[f'VaR.High [Return, 250, {risk}%]'],
             self.data[f'VaR.Relative [Return, 250, {risk}%]']) = self.build_var(self.data['Return'], period=250,
                                                                                 low_q=risk / 100,
                                                                                 high_q=1 - (risk / 100))
            (self.data[f'VaR.Low [Close, 250, {risk}%]'],
             self.data[f'VaR.High [Close, 250, {risk}%]'],
             self.data[f'VaR.Relative [Close, 250, {risk}%]']) = self.build_var(self.data['Close'], period=250,
                                                                                low_q=risk / 100,
                                                                                high_q=1 - (risk / 100))

        # Bollinger Bands
        for period in [10, 20, 30]:
            (self.data[f'UpperBand [{period}]'],
             self.data[f'LowerBand [{period}]'],
             self.data[f'BandRelLocation [{period}]']) = self.build_bollinger_bands(self.data["Close"], period=period)

        self.is_fitted = True
        return self.data