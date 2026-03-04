"""Start command handler — greeting and access request."""

from aiogram import Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

router = Router(name="start")


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Handle /start command — show greeting and access request button."""
    from bot.database.db import get_db
    from bot.config import load_config

    config = load_config()
    db = get_db()
    user = await db.get_user(message.from_user.id)

    if message.from_user.id == config.bot.admin_id:
        # Admin is always authorized
        if not user:
            await db.add_user(
                user_id=message.from_user.id,
                username=message.from_user.username,
                full_name=message.from_user.full_name,
                status="approved",
            )
        await message.answer(
            "👋 Привет, админ!\n\n"
            "Бот готов к работе. Отправь мне аудио или видео файл, "
            "и я расшифрую его в текст.\n\n"
            "📋 Команды:\n"
            "/users — список пользователей\n"
            "/stats — статистика\n"
        )
        return

    if user and user["status"] == "approved":
        await message.answer(
            "👋 С возвращением!\n\n"
            "Отправь мне аудио или видео файл, "
            "и я расшифрую его в текст.\n\n"
            "🎤 Поддерживаю: голосовые, видеосообщения, "
            "MP3, WAV, OGG, MP4 и другие форматы."
        )
    elif user and user["status"] == "pending":
        await message.answer(
            "⏳ Ваш запрос на доступ ожидает одобрения администратором.\n"
            "Пожалуйста, подождите."
        )
    elif user and user["status"] == "rejected":
        await message.answer(
            "❌ Ваш запрос на доступ был отклонён.\n"
            "Свяжитесь с администратором для получения доступа."
        )
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text="🔑 Запросить доступ",
                callback_data="request_access",
            )]
        ])
        await message.answer(
            "👋 Привет! Я бот для расшифровки аудио и видео в текст.\n\n"
            "🎤 Умею распознавать речь и разделять по спикерам.\n"
            "📝 Возвращаю красиво отформатированный текст.\n\n"
            "Для начала работы нажмите кнопку ниже, "
            "чтобы запросить доступ.",
            reply_markup=keyboard,
        )


@router.callback_query(F.data == "request_access")
async def on_request_access(callback: CallbackQuery) -> None:
    """Handle access request button press."""
    from bot.database.db import get_db
    from bot.config import load_config

    config = load_config()
    db = get_db()
    user = await db.get_user(callback.from_user.id)

    if user:
        await callback.answer("Вы уже отправляли запрос.", show_alert=True)
        return

    # Save user with pending status
    await db.add_user(
        user_id=callback.from_user.id,
        username=callback.from_user.username,
        full_name=callback.from_user.full_name,
        status="pending",
    )

    # Notify admin
    admin_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="✅ Одобрить",
                callback_data=f"approve_{callback.from_user.id}",
            ),
            InlineKeyboardButton(
                text="❌ Отклонить",
                callback_data=f"reject_{callback.from_user.id}",
            ),
        ]
    ])

    username_str = f"@{callback.from_user.username}" if callback.from_user.username else "нет"
    await callback.bot.send_message(
        chat_id=config.bot.admin_id,
        text=(
            "🔔 Новый запрос на доступ!\n\n"
            f"👤 Имя: {callback.from_user.full_name}\n"
            f"📛 Username: {username_str}\n"
            f"🆔 ID: {callback.from_user.id}"
        ),
        reply_markup=admin_keyboard,
    )

    await callback.answer()
    await callback.message.edit_text(
        "✅ Запрос на доступ отправлен!\n"
        "Ожидайте одобрения администратором."
    )


@router.callback_query(F.data.startswith("approve_"))
async def on_approve_user(callback: CallbackQuery) -> None:
    """Admin approves user access."""
    from bot.database.db import get_db
    from bot.config import load_config

    config = load_config()
    if callback.from_user.id != config.bot.admin_id:
        await callback.answer("Только админ может это делать.", show_alert=True)
        return

    user_id = int(callback.data.split("_")[1])
    db = get_db()
    await db.update_user_status(user_id, "approved")

    await callback.answer("Пользователь одобрен!")
    await callback.message.edit_text(
        callback.message.text + "\n\n✅ Доступ одобрен."
    )

    # Notify user
    try:
        await callback.bot.send_message(
            chat_id=user_id,
            text=(
                "🎉 Ваш доступ одобрен!\n\n"
                "Теперь вы можете отправлять мне аудио и видео файлы "
                "для расшифровки в текст.\n\n"
                "🎤 Поддерживаю: голосовые, видеосообщения, "
                "MP3, WAV, OGG, MP4 и другие форматы."
            ),
        )
    except Exception:
        pass  # User may have blocked the bot


@router.callback_query(F.data.startswith("reject_"))
async def on_reject_user(callback: CallbackQuery) -> None:
    """Admin rejects user access."""
    from bot.database.db import get_db
    from bot.config import load_config

    config = load_config()
    if callback.from_user.id != config.bot.admin_id:
        await callback.answer("Только админ может это делать.", show_alert=True)
        return

    user_id = int(callback.data.split("_")[1])
    db = get_db()
    await db.update_user_status(user_id, "rejected")

    await callback.answer("Пользователь отклонён.")
    await callback.message.edit_text(
        callback.message.text + "\n\n❌ Доступ отклонён."
    )

    # Notify user
    try:
        await callback.bot.send_message(
            chat_id=user_id,
            text="❌ К сожалению, ваш запрос на доступ был отклонён.",
        )
    except Exception:
        pass
