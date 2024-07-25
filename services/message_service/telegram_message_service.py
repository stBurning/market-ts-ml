import logging
import os
import time
from datetime import datetime
import g4f
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message, FSInputFile
from aiogram.enums import ParseMode
from aiogram import F

logging.basicConfig(
    level=logging.DEBUG,
    filename="logs.log",
    format="%(asctime)s - %(module)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s",
    datefmt='%H:%M:%S',
)

def ask_gpt(promt) -> str:
    response = g4f.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": promt}],
        stream=True,
    )

    ans_message = ''
    for message in response:
        ans_message += message

    return ans_message

formatter = logging.Formatter(fmt="%(asctime)s - %(module)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s", )

dp = Dispatcher()


ADMIN_ID = int(os.getenv('ADMIN_ID'))


class MessageBot:
    def __init__(self, token: str, admin_id: int):

        self.bot = Bot(token=token, default=DefaultBotProperties(parse_mode='HTML'))
        self.admin_id = admin_id
        # =============LOGS===============
        self.logger = logging.getLogger(self.__class__.__name__)
        cmd_handler = logging.StreamHandler()
        cmd_handler.setFormatter(formatter)
        self.logger.addHandler(cmd_handler)

    async def start(self):
        self.logger.info(msg="Bot is starting")
        dp.message.register(self.answer)

        try:
            await dp.start_polling(self.bot)
        finally:
            await self.bot.session.close()

    async def send_message(self, message):
        try:
            await self.bot.send_message(
                chat_id=self.admin_id,
                text=message,
                parse_mode=ParseMode.HTML
            )
        finally:
            self.logger.info(msg=f'Message "{message}" sent!')

    async def send_photo(self, path, message):
        try:
            await self.bot.send_photo(
                chat_id=self.admin_id,
                photo=FSInputFile(path),
                caption=message
            )
        finally:
            self.logger.info(msg=f'Photo with message "{message}" sent!')

    @staticmethod
    @dp.message(F.text)
    async def answer(message: Message):
        if message.from_user.id == ADMIN_ID:
            question = message.text
            answer = ask_gpt(question)
            await message.answer(answer, parse_mode=ParseMode.HTML)
            
        else:
            await message.answer(f"Привет, <b>{message.from_user.full_name}</b>! К сожалению, доступ для вас запрещен.", parse_mode=ParseMode.HTML)


async def at_minute_start(cb):
    """
    Функция откладывает выполнение передаваемой функции до начала следующей минуты
    :param cb: исходная функция
    :return:
    """
    while True:
        now = datetime.now()
        after_minute = now.second + now.microsecond / 1_000_000
        if after_minute:
            await asyncio.sleep(60 - after_minute)
        await cb()

async def at_five_minutes_start(cb):
    """
    Функция откладывает выполнение передаваемой функции до начала следующей минуты
    :param cb: исходная функция
    :return:
    """
    while True:
        now = datetime.now()
        minute = 5 - now.minute % 5
        after_minute = now.second + now.microsecond / 1_000_000
        if after_minute:
            print(f"Sleep {60 * minute - after_minute}s")
            await asyncio.sleep(60 * minute - after_minute)
        await cb()


if __name__ == '__main__':
    load_dotenv()
 
    API_TOKEN = os.getenv('API_TOKEN')
    ADMIN_ID = int(os.getenv('ADMIN_ID'))
    
    bot = MessageBot(API_TOKEN, ADMIN_ID)

    def get_message():
        message = f"""
            Текущее время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}         
        """
        return message


    async def run():
        await asyncio.gather(
            bot.start(),
            at_minute_start(
                lambda: bot.send_message(get_message())
            )
        )


    asyncio.run(run())