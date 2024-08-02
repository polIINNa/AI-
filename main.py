import os
import asyncio
import logging.config
from pathlib import Path

from dotenv import load_dotenv
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from bot.router import router


def init_dispatcher() -> Dispatcher:
    log_dir = Path(__file__).parent / 'log'
    if not log_dir.exists():
        log_dir.mkdir()
    logging.config.fileConfig(Path(__file__).parent / 'logging.conf',
                              {'log_file': log_dir / 'bot.log'})
    logger = logging.getLogger('bot_logger')
    dp = Dispatcher(logger=logger)
    dp.include_router(router)
    return dp


async def main() -> None:
    dp = init_dispatcher()
    token = os.getenv("TOKEN")
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
