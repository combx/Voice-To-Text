"""Handler for YouTube links — extracts subtitles and summarizes them."""

import logging
import os
import tempfile
from pathlib import Path

from aiogram import Router, F
from aiogram.types import Message, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton

from bot.services.auth import is_user_authorized
from bot.services.youtube import get_youtube_subs
from bot.services.formatter import format_with_llm
from bot.services.audio import format_duration, cleanup_temp_file, TEMP_DIR
from bot.services.transcriber import split_message, LOCALIZATIONS

logger = logging.getLogger(__name__)

router = Router(name="youtube")

# Regex for YouTube links
YOUTUBE_RE = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=)?[\w-]+'

@router.message(F.text.regexp(YOUTUBE_RE))
async def handle_youtube_link(message: Message) -> None:
    """Handle messages containing YouTube links."""
    if not await is_user_authorized(message):
        return

    # Use regexp search to find the URL in message text (it might contain other text)
    import re
    match = re.search(YOUTUBE_RE, message.text)
    if not match:
        return
    url = match.group(0)
    
    status_msg = await message.answer("🔍 Анализирую YouTube ссылку...")
    
    try:
        # Extract subtitles
        metadata = await get_youtube_subs(url, str(TEMP_DIR))
        
        if not metadata:
            await status_msg.edit_text("❌ Не удалось получить информацию о видео.")
            return

        if not metadata.subtitles_text:
            await status_msg.edit_text(
                f"📺 **{metadata.title}**\n\n"
                "❌ У этого видео нет доступных субтитров (ни созданных автором, ни автоматических).\n"
                "Бот пока умеет работать только с субтитрами."
            )
            return

        duration_str = format_duration(metadata.duration_seconds)
        
        await status_msg.edit_text(
            f"📥 Субтитры получены!\n"
            f"📺 **{metadata.title}**\n"
            f"⏱ Длительность: {duration_str}\n"
            f"🌐 Язык: {metadata.language or 'не определён'}\n"
            f"✨ Генерирую саммари..."
        )

        # Summarize subtitles via LLM
        lang_code = metadata.language or "en"
        loc = LOCALIZATIONS.get(lang_code, LOCALIZATIONS["en"]) if lang_code in LOCALIZATIONS else LOCALIZATIONS["en"]
        
        # Call LLM for summary only (we don't need formatting for raw subs, just summary)
        # We pass a large duration to trigger summary at the top
        llm_result = await format_with_llm(
            metadata.subtitles_text, 
            language=lang_code, 
            duration_seconds=metadata.duration_seconds
        )
        
        if llm_result.error:
            # Fallback to just sending subtitles if LLM fails
            summary = ""
            error_note = f"\n\n⚠️ Ошибка саммаризации: {llm_result.error}"
        else:
            summary = llm_result.formatted_text
            error_note = ""

        # Prepare response
        header = (
            f"📺 **{metadata.title}**\n"
            f"⏱ {loc['duration']}: {duration_str}\n"
            f"🌐 {loc['language']}: {metadata.language or '?'}\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
        )
        
        full_text = header + summary + error_note
        
        # Split and send summary
        parts = split_message(full_text)
        await status_msg.delete()
        
        for i, part in enumerate(parts):
            reply_markup = None
            if i == len(parts) - 1 and metadata.language != "ru":
                builder = InlineKeyboardBuilder()
                builder.row(InlineKeyboardButton(
                    text="🇷🇺 Перевести на русский",
                    callback_data=f"tr_yt_{metadata.video_id}"
                ))
                reply_markup = builder.as_markup()
            
            await message.answer(part, reply_markup=reply_markup)

        # Export full subtitles to .txt
        txt_path = str(TEMP_DIR / f"youtube_{metadata.video_id}.txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                # Add summary to the top of the file
                f.write(f"SUMMARY:\n{summary}\n\nFULL SUBTITLES:\n{metadata.subtitles_text}")
            
            doc = FSInputFile(txt_path, filename=f"YouTube_{metadata.video_id}_субтитры.txt")
            await message.answer_document(
                doc,
                caption="📄 Полные субтитры с саммари в начале"
            )
        finally:
            cleanup_temp_file(txt_path)
            # Also cleanup .vtt/.srt files if they were saved
            for ext in ['vtt', 'srt', 'ru.vtt', 'ru.srt', 'en.vtt', 'en.srt']:
                cleanup_temp_file(str(TEMP_DIR / f"{metadata.video_id}.{ext}"))

    except Exception as e:
        logger.exception("YouTube processing failed: %s", e)
        await status_msg.edit_text("❌ Произошла ошибка при обработке YouTube видео.")


# ─── YouTube Translation Callback Handler ───────────────────────────

from aiogram.types import CallbackQuery

@router.callback_query(F.data.startswith("tr_yt_"))
async def handle_yt_translate_callback(callback: CallbackQuery) -> None:
    """Handle 'Translate to Russian' button click for YouTube."""
    await callback.answer("⏳ Перевожу на русский...")
    status_msg = await callback.message.answer("✨ Перевожу текст на русский...")
    
    try:
        # Get subtitles from the original message (or rather the .txt file would be better, 
        # but here we can't easily get it. Let's assume we translate what's in the message if it's not too long,
        # otherwise we'd need to store subtitles in some state or DB.
        # However, for YouTube, we can just re-extract subs if it's quick, or translate the summary.
        # The user wants "перевод диалога на русский и саммори так же будет делаться на русском".
        # Since subs can be huge, translating the entire subs through LLM might be expensive/slow.
        # But let's try to get the original text from the message first.
        
        original_text = callback.message.text
        if "━━━━━━━━━━━━━━━━━━" in original_text:
            text_to_translate = original_text.split("━━━━━━━━━━━━━━━━━━")[-1].strip()
        else:
            text_to_translate = original_text

        # Call LLM with translation mode
        llm_result = await format_with_llm(text_to_translate, target_language="ru")
        
        if llm_result.error:
            await status_msg.edit_text(f"❌ Ошибка перевода: {llm_result.error}")
            return

        translated_header = "🇷🇺 **Перевод YouTube Sammary**\n━━━━━━━━━━━━━━━━━━\n"
        full_translated = translated_header + llm_result.formatted_text
        
        parts = split_message(full_translated)
        await status_msg.delete()
        for part in parts:
            await callback.message.answer(part)
            
    except Exception as e:
        logger.exception("YouTube translation failed: %s", e)
        await status_msg.edit_text("❌ Извините, не удалось выполнить перевод.")
