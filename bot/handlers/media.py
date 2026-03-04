"""Media file handlers — voice messages, video notes, and uploaded files."""

import logging
import os
import tempfile
from pathlib import Path

from aiogram import Router, F, Bot
from aiogram.types import Message

from bot.services.auth import is_user_authorized
from bot.services.audio import (
    extract_audio,
    get_audio_duration,
    format_duration,
    is_supported_format,
    get_file_type,
    cleanup_temp_file,
    TEMP_DIR,
)
from bot.config import load_config
from bot.database.db import get_db

logger = logging.getLogger(__name__)

router = Router(name="media")

import aiohttp
import asyncio
import time

async def _download_telegram_file(bot: Bot, file_id: str, suffix: str = ".ogg") -> tuple[str, float]:
    """Download a file from Telegram.
    
    Now uses a standard download approach as the network issue with 
    transparent proxies is resolved by switching to gRPC transport.
    """
    logger.info("Getting file info for file_id %s", file_id)
    file_info = await bot.get_file(file_id, request_timeout=30)
    file_path = str(TEMP_DIR / f"{file_id}{suffix}")
    
    t0 = time.monotonic()
    try:
        await bot.download(file=file_info, destination=file_path)
        actual = Path(file_path).stat().st_size
        if actual == 0:
            raise RuntimeError(f"Downloaded file is empty: {file_path}")
            
        elapsed = time.monotonic() - t0
        logger.info("Download complete: %s (%d bytes, %.1fs)", file_path, actual, elapsed)
    except Exception as e:
        logger.exception("Download failed: %s", e)
        raise

    return file_path, elapsed


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} Б"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} КБ"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} МБ"


async def _process_media(message: Message, input_path: str, file_name: str, file_type: str, file_size_bytes: int = 0, download_time: float = 0) -> None:
    """Common processing pipeline for all media types.

    Downloads → extracts audio → saves metadata → sends status updates.
    Currently stops after audio extraction (AssemblyAI integration in Этап 3).
    """
    config = load_config()
    db = get_db()
    wav_path = None

    try:
        # Status: Checking duration
        duration = await get_audio_duration(input_path)
        duration_str = format_duration(duration) if duration > 0 else "неизвестно"

        # Check duration limit
        if duration > config.app.max_audio_duration:
            max_dur = format_duration(config.app.max_audio_duration)
            await message.answer(
                f"❌ Файл слишком длинный ({duration_str}).\n"
                f"Максимальная длительность: {max_dur}"
            )
            return

        # Check balance before processing
        balance = await db.get_balance(
            initial_balance=config.app.assemblyai_initial_balance,
            rate_per_hour=config.app.assemblyai_rate_per_hour,
        )
        est_cost = (duration / 3600) * config.app.assemblyai_rate_per_hour
        if balance["remaining"] < est_cost:
            await message.answer(
                f"❌ Недостаточно баланса для обработки.\n"
                f"💰 Баланс: ${balance['remaining']:.2f}\n"
                f"💸 Ожидаемая стоимость: ${est_cost:.2f}\n"
                f"Пополните баланс на AssemblyAI."
            )
            return

        # Status: Preparing audio
        size_str = format_file_size(file_size_bytes) if file_size_bytes else "неизвестно"
        dl_str = f"{download_time:.1f} сек" if download_time else "—"
        await message.answer(
            f"🔄 Подготавливаю аудио...\n"
            f"📁 {file_name}\n"
            f"💾 Размер: {size_str}\n"
            f"⬇️ Загрузка: {dl_str}\n"
            f"⏱ Длительность: {duration_str}"
        )

        # Extract audio to WAV
        wav_path = await extract_audio(input_path)

        # Record in database
        transcription_id = await db.add_transcription(
            user_id=message.from_user.id,
            file_name=file_name,
            file_type=file_type,
            duration_seconds=duration,
        )

        # Status: Transcribing (with balance info)
        await message.answer(
            f"🎙 Распознаю речь...\n"
            f"📁 {file_name}\n"
            f"💾 Размер: {size_str}\n"
            f"⬇️ Загрузка: {dl_str}\n"
            f"⏱ Длительность: {duration_str}\n"
            f"💰 Баланс: ${balance['remaining']:.2f} (~{balance['hours_remaining']:.0f}ч)\n"
            f"📝 ID задачи: {transcription_id}"
        )

        # Heartbeat: periodic updates during long processing
        heartbeat_active = True
        heartbeat_counter = [0]

        async def _heartbeat(stage: str):
            """Send periodic status updates every 30 seconds."""
            while heartbeat_active:
                await asyncio.sleep(30)
                if not heartbeat_active:
                    break
                heartbeat_counter[0] += 1
                elapsed = heartbeat_counter[0] * 30
                try:
                    await message.answer(
                        f"⏳ {stage}... ({elapsed}с)\n"
                        f"📁 {file_name}"
                    )
                except Exception:
                    pass

        # Transcribe via AssemblyAI (with heartbeat)
        from bot.services.transcriber import transcribe_audio, format_transcription, split_message
        from bot.services.formatter import format_with_llm

        heartbeat_task = asyncio.create_task(_heartbeat("Распознаю речь"))
        try:
            result = await transcribe_audio(wav_path)
        finally:
            heartbeat_active = False
            heartbeat_task.cancel()

        # Update database
        await db.complete_transcription(
            transcription_id,
            language=result.language,
            speakers_count=result.speakers_count,
        )

        # LLM formatting (if OpenRouter key is configured)
        # Build speaker-labeled text for LLM (with emojis and timestamps)
        from bot.services.transcriber import _speaker_emoji, _format_timestamp
        
        if result.utterances:
            speaker_map = {}
            text_parts = []
            for u in result.utterances:
                emoji = _speaker_emoji(u.speaker, speaker_map)
                ts = _format_timestamp(u.start_ms)
                speaker_num = list(speaker_map.keys()).index(u.speaker) + 1
                text_parts.append(f"{emoji} Спикер {speaker_num} [{ts}]\n{u.text}")
            text_for_llm = "\n\n".join(text_parts)
        else:
            text_for_llm = result.text

        llm_result = None
        if text_for_llm and len(text_for_llm.strip()) > 10:
            await message.answer(
                f"✨ Форматирую текст...\n"
                f"📁 {file_name}\n"
                f"⏱ Длительность: {duration_str}"
            )
            # Heartbeat for LLM formatting
            heartbeat_active = True
            heartbeat_counter[0] = 0
            heartbeat_task = asyncio.create_task(_heartbeat("Форматирую"))
            try:
                llm_result = await format_with_llm(text_for_llm, result.language)
            finally:
                heartbeat_active = False
                heartbeat_task.cancel()

        # Format and send result
        formatted = format_transcription(
            result, file_name, duration_str,
            llm_text=llm_result.formatted_text if llm_result and not llm_result.error else None,
            llm_model=llm_result.model_used if llm_result and not llm_result.error else None,
        )
        parts = split_message(formatted)

        for part in parts:
            await message.answer(part)

        # Export to .txt for long transcriptions
        if len(formatted) > 4000:
            txt_path = str(TEMP_DIR / f"transcription_{transcription_id}.txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(formatted)
                from aiogram.types import FSInputFile
                doc = FSInputFile(txt_path, filename=f"{file_name}_расшифровка.txt")
                await message.answer_document(
                    doc,
                    caption="📄 Полная расшифровка в текстовом файле"
                )
            finally:
                cleanup_temp_file(txt_path)

        logger.info(
            "Transcription done: user=%s, file=%s, type=%s, duration=%.1fs, "
            "speakers=%d, lang=%s, llm=%s, task=%d",
            message.from_user.id, file_name, file_type, duration,
            result.speakers_count, result.language,
            llm_result.model_used if llm_result else "none",
            transcription_id,
        )

    except RuntimeError as e:
        logger.error("Audio processing error: %s", e)
        if 'transcription_id' in locals():
            await db.fail_transcription(transcription_id)
        await message.answer(
            f"❌ Ошибка при обработке файла:\n{str(e)[:200]}"
        )
    except Exception as e:
        logger.exception("Unexpected error processing media: %s", e)
        if 'transcription_id' in locals():
            await db.fail_transcription(transcription_id)
        await message.answer(
            "❌ Произошла непредвиденная ошибка при обработке файла.\n"
            "Попробуйте ещё раз или обратитесь к администратору."
        )
    finally:
        # Clean up temp files
        cleanup_temp_file(input_path)
        if wav_path:
            cleanup_temp_file(wav_path)


# ─── Voice messages ─────────────────────────────────────────────────

@router.message(F.voice)
async def handle_voice(message: Message) -> None:
    """Handle voice messages (audio notes)."""
    if not await is_user_authorized(message):
        return

    file_size = message.voice.file_size or 0
    size_str = format_file_size(file_size) if file_size else ""
    await message.answer(f"⏳ Голосовое сообщение получено ({size_str}), обрабатываю...")

    try:
        input_path, dl_time = await _download_telegram_file(
            message.bot, message.voice.file_id, suffix=".ogg"
        )
        file_name = "Голосовое сообщение"
        await _process_media(message, input_path, file_name, "voice", file_size, dl_time)
    except Exception as e:
        logger.error("FATAL ERROR in handle_voice: %s", e, exc_info=True)
        await message.answer("❌ Произошла ошибка при скачивании голосового сообщения.")


# ─── Video notes (кружочки) ─────────────────────────────────────────

@router.message(F.video_note)
async def handle_video_note(message: Message) -> None:
    """Handle video notes (circle videos)."""
    if not await is_user_authorized(message):
        return

    file_size = message.video_note.file_size or 0
    size_str = format_file_size(file_size) if file_size else ""
    await message.answer(f"⏳ Видеосообщение получено ({size_str}), обрабатываю...")

    try:
        input_path, dl_time = await _download_telegram_file(
            message.bot, message.video_note.file_id, suffix=".mp4"
        )
        file_name = "Видеосообщение (кружок)"
        await _process_media(message, input_path, file_name, "video_note", file_size, dl_time)
    except Exception as e:
        logger.error("FATAL ERROR in handle_video_note: %s", e, exc_info=True)
        await message.answer("❌ Произошла ошибка при скачивании кружочка.")


# ─── Video messages ──────────────────────────────────────────────────

@router.message(F.video)
async def handle_video(message: Message) -> None:
    """Handle video files sent as video messages."""
    if not await is_user_authorized(message):
        return

    config = load_config()
    if message.video.file_size and message.video.file_size > config.app.max_file_size:
        max_mb = config.app.max_file_size / (1024 * 1024)
        await message.answer(f"❌ Файл слишком большой. Максимум: {max_mb:.0f} МБ")
        return

    file_size = message.video.file_size or 0
    size_str = format_file_size(file_size) if file_size else ""
    await message.answer(f"⏳ Видео получено ({size_str}), обрабатываю...")

    try:
        file_name = message.video.file_name or "video.mp4"
        input_path, dl_time = await _download_telegram_file(
            message.bot, message.video.file_id, suffix=Path(file_name).suffix or ".mp4"
        )
        await _process_media(message, input_path, file_name, "video", file_size, dl_time)
    except Exception as e:
        logger.error("FATAL ERROR in handle_video: %s", e, exc_info=True)
        await message.answer("❌ Произошла ошибка при скачивании видео.")


# ─── Audio files ─────────────────────────────────────────────────────

@router.message(F.audio)
async def handle_audio(message: Message) -> None:
    """Handle audio files (music, podcasts, etc.)."""
    if not await is_user_authorized(message):
        return

    config = load_config()
    if message.audio.file_size and message.audio.file_size > config.app.max_file_size:
        max_mb = config.app.max_file_size / (1024 * 1024)
        await message.answer(f"❌ Файл слишком большой. Максимум: {max_mb:.0f} МБ")
        return

    file_size = message.audio.file_size or 0
    size_str = format_file_size(file_size) if file_size else ""
    await message.answer(f"⏳ Аудиофайл получен ({size_str}), обрабатываю...")

    try:
        file_name = message.audio.file_name or "audio.mp3"
        input_path, dl_time = await _download_telegram_file(
            message.bot, message.audio.file_id, suffix=Path(file_name).suffix or ".mp3"
        )
        await _process_media(message, input_path, file_name, "audio", file_size, dl_time)
    except Exception as e:
        logger.error("FATAL ERROR in handle_audio: %s", e, exc_info=True)
        await message.answer("❌ Произошла ошибка при скачивании аудиофайла.")


# ─── Documents (uploaded files) ──────────────────────────────────────

@router.message(F.document)
async def handle_document(message: Message) -> None:
    """Handle uploaded documents — only process supported audio/video formats."""
    if not await is_user_authorized(message):
        return

    file_name = message.document.file_name or "file"

    if not is_supported_format(file_name):
        supported = ", ".join(sorted(
            ext.lstrip(".").upper()
            for ext in sorted(
                list({".mp3", ".wav", ".ogg", ".flac", ".m4a", ".mp4", ".avi", ".mkv", ".mov", ".webm"})
            )
        ))
        await message.answer(
            f"❌ Формат файла не поддерживается.\n\n"
            f"Поддерживаемые форматы:\n{supported}"
        )
        return

    config = load_config()
    if message.document.file_size and message.document.file_size > config.app.max_file_size:
        max_mb = config.app.max_file_size / (1024 * 1024)
        await message.answer(f"❌ Файл слишком большой. Максимум: {max_mb:.0f} МБ")
        return

    file_size = message.document.file_size or 0
    size_str = format_file_size(file_size) if file_size else ""
    await message.answer(f"⏳ Файл «{file_name}» получен ({size_str}), обрабатываю...")

    try:
        file_type = get_file_type(file_name)
        input_path, dl_time = await _download_telegram_file(
            message.bot, message.document.file_id, suffix=Path(file_name).suffix
        )
        await _process_media(message, input_path, file_name, file_type, file_size, dl_time)
    except Exception as e:
        logger.error("FATAL ERROR in handle_document: %s", e, exc_info=True)
        await message.answer("❌ Произошла ошибка при скачивании файла.")
