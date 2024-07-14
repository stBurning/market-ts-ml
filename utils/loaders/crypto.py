import logging
import os
from datetime import datetime
from typing import Literal, Optional

import pandas as pd
from dotenv.main import load_dotenv
from pybit.exceptions import InvalidRequestError
from pybit.unified_trading import HTTP
from binance.client import Client
from binance.exceptions import BinanceAPIException
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def get_klines(self, *args, **kwargs) -> pd.DataFrame:
        pass



logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(lineno)d - %(message)s")



class BinanceLoader(BaseLoader):
    def __init__(self, api_key: str, secret_key: str):
        """
        BinanceLoader - class for getting data from Binance
        :param api_key: public API key
        :param secret_key: private API key
        """
        self.client = Client(api_key, secret_key)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())

    def get_klines(self, symbol: str,
                   interval=Client.KLINE_INTERVAL_5MINUTE,
                   duration: str = '1h UTC') -> Optional[pd.DataFrame]:
        """
        Method for getting data about price such as Open, Close, High and Low prices
        :param symbol: name of a trade pair like BTCUSDT
        :param interval: time fraction
        :param duration: time duration
        :return: pd.DataFrame with data or None
        """
        try:
            data = pd.DataFrame(self.client.get_historical_klines(symbol, interval, duration))
            data = data.iloc[:, :-1]
            data.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                            'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
                            'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume']
            data = data.set_index('Time')
            data.index = pd.to_datetime(data.index, unit='ms')
            data = data.astype(float)
        except BinanceAPIException as e:
            self.logger.critical(e.message)
            return None
        except Exception as e:
            self.logger.error(e)
            return None
        return data

    def get_order_book(self, symbol: str):
        data = pd.DataFrame(self.client.get_order_book(symbol=symbol)['bids'])
        data.columns = ['Price', 'Value']
        return data


class BybitLoader(BaseLoader):
    def __init__(self, api_key=None, secret_key=None):
        """
        BybitLoader - class for getting data from Bybit
        :param api_key: public API key
        :param secret_key: private API key
        """

        self.conn = HTTP(testnet=False, api_key=api_key, api_secret=secret_key)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())

    def get_klines(self, symbol: str,
                   interval: int = 5,
                   category: Literal["spot", "linear", "inverse"] = "spot",
                   start: float = None,
                   end: float = datetime.now().timestamp()):
        """
        Method for getting data about price such as Open, Close, High and Low prices

        :param category: Literal["spot", "linear", "inverse"]
        :param symbol: name of a trade pair like BTCUSDT
        :param interval: time fraction
        :param start: start timestamp
        :param end: end timestamp

        :return: pd.DataFrame with data
        """
        try:
            kline = self.conn.get_kline(symbol=symbol,
                                        cathegory=category,
                                        interval=interval,
                                        start=start,
                                        end=end)

            result = pd.DataFrame(kline['result']['list'])
            result.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
            result['Time'] = pd.to_datetime(result['Time'].astype("int64"), unit='ms')
            result.set_index('Time', drop=True, inplace=True)
        except InvalidRequestError as e:
            self.logger.critical(e.message)
            return None
        except Exception as e:
            self.logger.error(e)
            return None
        return result

    def get_order_book(self, symbol):
        raise NotImplementedError()
