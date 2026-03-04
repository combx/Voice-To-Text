"""Authorization middleware — checks if user is approved before processing media."""

from aiogram.types import Message

from bot.config import load_config
from bot.database.db import get_db


async def is_user_authorized(message: Message) -> bool:
    """Check if the user is authorized to use the bot.

    Returns True for admin and approved users.
    """
    config = load_config()

    # Admin is always authorized
    if message.from_user.id == config.bot.admin_id:
        return True

    db = get_db()
    user = await db.get_user(message.from_user.id)

    if user and user["status"] == "approved":
        return True

    # Notify unauthorized user
    if user and user["status"] == "pending":
        await message.answer(
            "⏳ Ваш запрос на доступ ожидает одобрения.\n"
            "Пожалуйста, подождите."
        )
    elif user and user["status"] == "rejected":
        await message.answer("❌ У вас нет доступа к боту.")
    else:
        await message.answer(
            "🔒 Для использования бота нужен доступ.\n"
            "Нажмите /start чтобы запросить доступ."
        )

    return False
