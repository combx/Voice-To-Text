"""Admin command handlers."""

from aiogram import Router
from aiogram.filters import Command, BaseFilter
from aiogram.types import Message
from bot.config import load_config

class IsAdminFilter(BaseFilter):
    async def __call__(self, message: Message, **kwargs) -> bool:
        config = kwargs.get("config") or load_config()
        return message.from_user.id == config.bot.admin_id

router = Router(name="admin")
router.message.filter(IsAdminFilter())


@router.message(Command("users"))
async def cmd_users(message: Message) -> None:
    """Show list of all users."""
    from bot.database.db import get_db

    db = get_db()
    users = await db.get_all_users()

    if not users:
        await message.answer("📋 Пользователей пока нет.")
        return

    status_emoji = {
        "approved": "✅",
        "pending": "⏳",
        "rejected": "❌",
    }

    lines = ["📋 <b>Список пользователей:</b>\n"]
    for user in users:
        emoji = status_emoji.get(user["status"], "❓")
        username = f"@{user['username']}" if user["username"] else "—"
        lines.append(
            f"{emoji} {user['full_name']} ({username})\n"
            f"   ID: <code>{user['user_id']}</code> | "
            f"Статус: {user['status']}"
        )

    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("approve"))
async def cmd_approve(message: Message) -> None:
    """Approve user by ID: /approve <user_id>."""
    from bot.database.db import get_db

    parts = message.text.split()
    if len(parts) != 2:
        await message.answer("Использование: /approve <user_id>")
        return

    try:
        user_id = int(parts[1])
    except ValueError:
        await message.answer("❌ Некорректный ID пользователя.")
        return

    db = get_db()
    user = await db.get_user(user_id)
    if not user:
        await message.answer("❌ Пользователь не найден.")
        return

    await db.update_user_status(user_id, "approved")
    await message.answer(f"✅ Пользователь {user['full_name']} одобрен.")

    try:
        await message.bot.send_message(
            chat_id=user_id,
            text="🎉 Ваш доступ одобрен! Можете отправлять файлы для расшифровки.",
        )
    except Exception:
        pass


@router.message(Command("revoke"))
async def cmd_revoke(message: Message) -> None:
    """Revoke user access: /revoke <user_id>."""
    from bot.database.db import get_db

    parts = message.text.split()
    if len(parts) != 2:
        await message.answer("Использование: /revoke <user_id>")
        return

    try:
        user_id = int(parts[1])
    except ValueError:
        await message.answer("❌ Некорректный ID пользователя.")
        return

    db = get_db()
    user = await db.get_user(user_id)
    if not user:
        await message.answer("❌ Пользователь не найден.")
        return

    await db.update_user_status(user_id, "rejected")
    await message.answer(f"🚫 Доступ пользователя {user['full_name']} отозван.")


@router.message(Command("stats"))
async def cmd_stats(message: Message, **kwargs) -> None:
    """Show bot usage statistics."""
    from bot.database.db import get_db

    db = get_db()
    stats = await db.get_stats()
    
    config = kwargs.get("config") or load_config()
    balance = await db.get_balance(
        initial_balance=config.app.assemblyai_initial_balance,
        rate_per_hour=config.app.assemblyai_rate_per_hour,
    )
    
    or_balance_text = "Не настроен"
    if config.openrouter.api_key:
        from bot.services.formatter import get_openrouter_balance
        or_bal = await get_openrouter_balance(config.openrouter.api_key)
        or_balance_text = f"${or_bal['remaining']:.2f}"

    await message.answer(
        "📊 <b>Статистика бота:</b>\n\n"
        f"👥 Пользователей: {stats['total_users']}\n"
        f"   ✅ Одобрено: {stats['approved_users']}\n"
        f"   ⏳ Ожидают: {stats['pending_users']}\n"
        f"   ❌ Отклонено: {stats['rejected_users']}\n\n"
        f"📝 Расшифровок: {stats['total_transcriptions']}\n"
        f"⏱ Обработано аудио: {stats['total_duration_formatted']}\n\n"
        f"💰 <b>Баланс API:</b>\n"
        f"🎙 Звук (AssemblyAI): ${balance['remaining']:.2f} (~{balance['hours_remaining']:.0f}ч)\n"
        f"🤖 Текст (OpenRouter): {or_balance_text}",
        parse_mode="HTML",
    )
