"""Speech-to-text transcription service using AssemblyAI REST API.

Uses httpx for HTTP calls (supports explicit write timeouts needed for uploads).
Supports automatic language detection and speaker diarization.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from bot.config import load_config

logger = logging.getLogger(__name__)

# Speaker color emojis for Telegram output
SPEAKER_COLORS = ["🔵", "🟢", "🔴", "🟡", "🟣", "🟤", "⚪", "🟠"]

ASSEMBLYAI_BASE = "https://api.assemblyai.com/v2"


@dataclass
class Utterance:
    """A single speaker utterance."""
    speaker: str       # e.g. "A", "B"
    text: str
    start_ms: int
    end_ms: int


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    utterances: list[Utterance] = field(default_factory=list)
    language: str = ""
    speakers_count: int = 0
    confidence: float = 0.0
    error: str = ""


def _format_timestamp(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    total_sec = ms // 1000
    minutes = total_sec // 60
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:02d}"


def _speaker_emoji(speaker: str, speaker_map: dict) -> str:
    """Get a color emoji for a speaker, assigning new colors as needed."""
    if speaker not in speaker_map:
        idx = len(speaker_map) % len(SPEAKER_COLORS)
        speaker_map[speaker] = SPEAKER_COLORS[idx]
    return speaker_map[speaker]


def _sync_upload_file(api_key: str, wav_path: str) -> str:
    """Upload audio file to AssemblyAI (synchronous, run in thread).

    Uses httpx with generous write/read timeouts.
    Returns the upload URL.
    """
    import os
    file_size = os.path.getsize(wav_path)
    logger.info("Uploading %s (%d bytes) to AssemblyAI...", wav_path, file_size)

    headers = {
        "authorization": api_key,
        "content-type": "application/octet-stream",
    }

    with open(wav_path, "rb") as f:
        file_data = f.read()

    timeout = httpx.Timeout(timeout=300.0, connect=30.0, read=300.0, write=300.0)

    t0 = time.monotonic()
    
    max_retries = 3
    resp = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout, http2=True) as client:
                resp = client.post(
                    f"{ASSEMBLYAI_BASE}/upload",
                    headers=headers,
                    content=file_data,
                )
            break
        except httpx.RequestError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Сетевая ошибка при загрузке аудио в AssemblyAI после {max_retries} попыток: {e}") from e
            logger.warning("Upload attempt %d failed: %s. Retrying in 2 seconds...", attempt + 1, e)
            import time as _time
            _time.sleep(2)

    elapsed = time.monotonic() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"AssemblyAI upload failed ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    upload_url = data["upload_url"]
    logger.info("File uploaded to AssemblyAI in %.1fs: %s", elapsed, upload_url)
    return upload_url


def _sync_create_transcript(api_key: str, audio_url: str) -> str:
    """Submit transcription request (synchronous). Returns transcript ID."""
    headers = {
        "authorization": api_key,
        "content-type": "application/json",
    }
    payload = {
        "audio_url": audio_url,
        "speech_models": ["universal-2"],
        "speaker_labels": True,
        "language_detection": True,
    }

    timeout = httpx.Timeout(timeout=30.0)
    with httpx.Client(timeout=timeout, http2=True) as client:
        resp = client.post(
            f"{ASSEMBLYAI_BASE}/transcript",
            headers=headers,
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"AssemblyAI transcript request failed ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    transcript_id = data["id"]
    logger.info("Transcript requested: %s", transcript_id)
    return transcript_id


def _sync_poll_transcript(api_key: str, transcript_id: str) -> dict:
    """Poll until transcription is complete (synchronous). Returns full JSON."""
    headers = {"authorization": api_key}
    url = f"{ASSEMBLYAI_BASE}/transcript/{transcript_id}"
    timeout = httpx.Timeout(timeout=30.0)

    while True:
        with httpx.Client(timeout=timeout, http2=True) as client:
            resp = client.get(url, headers=headers)

        data = resp.json()
        status = data["status"]

        if status == "completed":
            logger.info("Transcript %s completed", transcript_id)
            return data
        elif status == "error":
            error = data.get("error", "Unknown error")
            raise RuntimeError(f"Ошибка распознавания: {error}")
        else:
            logger.debug("Transcript %s status: %s", transcript_id, status)
            import time as _time
            _time.sleep(3)


async def transcribe_audio(wav_path: str) -> TranscriptionResult:
    """Transcribe an audio file using AssemblyAI REST API.
    
    All HTTP calls use httpx (synchronous) run in a thread pool
    to avoid blocking the event loop while being resilient to proxy issues.
    """
    config = load_config()
    api_key = config.assemblyai.api_key

    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY не настроен")

    # Step 1: Upload (run in thread to avoid blocking event loop)
    upload_url = await asyncio.to_thread(_sync_upload_file, api_key, wav_path)

    # Step 2: Create transcript
    transcript_id = await asyncio.to_thread(_sync_create_transcript, api_key, upload_url)

    # Step 3: Poll for result (blocking sleep inside, so run in thread)
    data = await asyncio.to_thread(_sync_poll_transcript, api_key, transcript_id)

    # Step 4: Parse result
    utterances = []
    if data.get("utterances"):
        for u in data["utterances"]:
            utterances.append(Utterance(
                speaker=u["speaker"],
                text=u["text"],
                start_ms=u["start"],
                end_ms=u["end"],
            ))

    speakers = {u.speaker for u in utterances}

    result = TranscriptionResult(
        text=data.get("text", ""),
        utterances=utterances,
        language=data.get("language_code", ""),
        speakers_count=len(speakers),
        confidence=data.get("confidence", 0),
    )

    return result


def format_transcription(
    result: TranscriptionResult,
    file_name: str,
    duration_str: str,
    llm_text: str | None = None,
    llm_model: str | None = None,
) -> str:
    """Format transcription result for Telegram message.
    
    If llm_text is provided, uses LLM-formatted text instead of raw utterances.
    """
    lang_name = _get_language_name(result.language)
    
    header_lines = [
        "📝 Расшифровка аудио",
        "━━━━━━━━━━━━━━━━━━",
        f"📁 Файл: {file_name}",
        f"⏱ Длительность: {duration_str}",
        f"🗣 Спикеров: {result.speakers_count}",
        f"🌐 Язык: {lang_name}",
    ]
    if llm_model and llm_model != "none":
        model_short = llm_model.split("/")[-1].split(":")[0]
        header_lines.append(f"🤖 Формат: {model_short}")
    header_lines.append("━━━━━━━━━━━━━━━━━━\n")
    header = "\n".join(header_lines) + "\n"

    # Use LLM-formatted text if available
    if llm_text:
        body = llm_text + "\n"
    elif result.utterances:
        speaker_map = {}
        body_parts = []
        for u in result.utterances:
            emoji = _speaker_emoji(u.speaker, speaker_map)
            ts = _format_timestamp(u.start_ms)
            speaker_num = list(speaker_map.keys()).index(u.speaker) + 1
            body_parts.append(f"{emoji} Спикер {speaker_num} [{ts}]\n{u.text}\n")
        body = "\n".join(body_parts)
    else:
        body = result.text + "\n"

    return header + body


def split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= max_len:
        return [text]

    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, max_len)
        if cut <= 0:
            cut = max_len
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")

    return parts


LANGUAGE_MAP = {
    "ru": "Русский",
    "en": "English",
    "uk": "Українська",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "pl": "Polski",
    "ja": "日本語",
    "zh": "中文",
    "ko": "한국어",
    "tr": "Türkçe",
    "ar": "العربية",
    "hi": "हिन्दी",
}


def _get_language_name(code: str) -> str:
    """Convert language code to human-readable name."""
    if not code:
        return "Не определён"
    return LANGUAGE_MAP.get(code, code)
