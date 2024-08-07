from logging import Logger

from aiogram.types import Message
from aiogram import Router, html, Bot
from aiogram.filters import CommandStart

from agent.cyqiq import get_answer

router = Router()


@router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    Обработчик для команды `/start`
    """
    await message.answer(f"Имя мне Табличный гуру.\n"
                         f"Приветствую, {html.bold(message.from_user.full_name)}.\n"
                         f"Задавай свой вопрос или проваливай.")


@router.message()
async def question_handler(message: Message, bot: Bot, logger: Logger) -> None:
    """
    Обработчик вопроса пользователя

    :param message:     Входное сообщение
    :param bot:         Инстанс бота
    :param logger:      Логгер
    """
    try:
        logger.info(f'Обработка вопроса от пользователя text={message.text}')
        answer = get_answer(message.text, logger)
        if answer is None:
            await message.answer(text="Не найти ответа..")
        await message.answer(text=answer)
    except Exception as error:
        await message.answer(text="Возникла ошибка при поиске ответа.")
        logger.exception(f"Произошла ошибка при обработке вопроса: {message.text}", exc_info=error)
