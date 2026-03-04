"""Main entry point for the VoiceToText Telegram bot."""

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import load_config
from bot.database.db import get_db
from bot.handlers import start, admin


async def main() -> None:
    """Initialize and start the bot."""
    config = load_config()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.app.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    # Initialize database
    db = get_db()
    await db.init()
    logger.info("Database initialized")

    # Create bot and dispatcher
    bot = Bot(
        token=config.bot.token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    # Register routers
    dp.include_router(start.router)
    dp.include_router(admin.router)

    # Start polling
    logger.info("Bot starting...")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
