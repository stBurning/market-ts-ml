import os
from datetime import datetime
import asyncio
import pandas as pd
from dotenv.main import load_dotenv
from model_service.model_service import make_predict
from load_service.load_service import load_data
from message_service.telegram_message_service import MessageBot, at_minute_start, at_five_minutes_start
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":

    logger.info("Loading Envs")
    load_dotenv()

    API_TOKEN = os.environ['API_TOKEN']
    ADMIN_ID = int(os.environ['ADMIN_ID'])
    IMG_SAVE_PATH = os.environ['IMG_SAVE_PATH']

    def get_marker(value):
        return '\U0001F7E2' if value > 0.5 else '\U0001F534'

    def get_prediction_result():
        prediction_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = load_data("BTCUSDT")
        predict = make_predict(df)
        messages = [
            "\U0000231B" + f" <b>Начало вычислений</b>: {prediction_time}",
            "\U00002705" + f" <b>Текущее время</b>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            get_marker(predict) + f"<b>Предсказание</b>: {predict:.4f}"
        ]
        
        return "\n".join(messages)  

    bot = MessageBot(API_TOKEN, ADMIN_ID)


    async def run():
        await asyncio.gather(
            bot.start(),
            at_five_minutes_start(
                lambda: bot.send_photo(IMG_SAVE_PATH, get_prediction_result())
            )
        )


    asyncio.run(run())