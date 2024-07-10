import logging
import os
from datetime import datetime
from typing import Literal

import pandas as pd
from dotenv.main import load_dotenv
from pybit.exceptions import InvalidRequestError
from pybit.unified_trading import HTTP

from BaseLoader import BaseLoader

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(lineno)d - %(message)s")


class BybitLoader(BaseLoader):
    def __init__(self, api_key=None, secret_key=None):
        """
        BybitLoader - class for getting data from Bybit
        :param api_key: public API key
        :param secret_key: private API key
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.conn = HTTP()
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


if __name__ == "__main__":

    load_dotenv()

    DATA_PATH = os.environ['BYBIT_LOADING_SERVICE_PATH']

    loader = BybitLoader()

    df = loader.get_klines(symbol="BTCUSDT", interval=60, start=datetime(2024, 1, 1).timestamp())

    if df is not None:
        df.to_csv(os.path.join(DATA_PATH, datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"))
