from datetime import datetime
from typing import Union
import pandas as pd
import yfinance as yf



def yfinance_load(ticker: str,
                  start_point: Union[datetime, str], end_point: datetime = datetime.now(),
                  interval: str = '1d') -> pd.DataFrame:
    data: pd.DataFrame = yf.download(tickers=ticker, start=start_point, end=end_point, interval=interval)
    return data