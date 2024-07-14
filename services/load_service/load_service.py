import os
import logging
from dotenv.main import load_dotenv
from utils.loaders.crypto import BybitLoader, BinanceLoader



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def load_data(ticker="BTCUSDT"):
    logger.info("Loading Data")
    load_dotenv() # Text yout API-keys in .env 
    
    BINANCE_API_KEY = os.environ["BINANCE_API_KEY"]
    BINANCE_SECRET_KEY = os.environ["BINANCE_SECRET_KEY"]


    loader = BinanceLoader(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    data = loader.get_klines(ticker, interval="5m", duration="10 hours UTC")
    logger.info("Data successfuly loaded")
    return data
