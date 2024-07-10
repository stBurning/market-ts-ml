import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv.main import load_dotenv

from BaseLoader import BaseLoader

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


if __name__ == "__main__":

    load_dotenv()

    API_KEY = os.environ['BINANCE_API_KEY']
    SECRET_KEY = os.environ['BINANCE_SECRET_KEY']
    DATA_PATH = os.environ['BINANCE_LOADING_SERVICE_PATH']

    loader = BinanceLoader(API_KEY, SECRET_KEY)

    df = loader.get_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, duration="30 days UTC")

    if df is not None:
        df.to_csv(os.path.join(DATA_PATH, datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"))
