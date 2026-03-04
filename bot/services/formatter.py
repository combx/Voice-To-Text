"""Text formatting service using OpenRouter LLM API.

Formats raw transcription text: adds punctuation, paragraphs, and a summary.
CRITICAL: Never removes or changes any words — text must remain verbatim.
Uses httpx with HTTP/2 and a fallback chain of free models.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from bot.config import load_config

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """Ты — профессиональный редактор-форматировщик расшифровок аудиозаписей.

СТРОГИЕ ПРАВИЛА:
1. НЕ УДАЛЯЙ ни одного слова из текста. Каждое слово должно остаться на месте.
2. НЕ ЗАМЕНЯЙ слова на синонимы. НЕ перефразируй.
3. НЕ ДОБАВЛЯЙ новых слов (кроме знаков препинания).
4. СОХРАНЯЙ все метки спикеров (🔵 Спикер 1, 🟢 Спикер 2 и т.д.) и временные метки [MM:SS] в точности как в оригинале.
5. РАЗРЕШЕНО только:
   - Расставить знаки препинания (запятые, точки, вопросительные и восклицательные знаки, тире, двоеточия)
   - Расставить заглавные буквы в начале предложений и для имён собственных
   - Разбить длинные реплики одного спикера на абзацы по смыслу

ФОРМАТ ОТВЕТА:
Сначала выведи отформатированный текст с СОХРАНЕНИЕМ всех меток спикеров и временных меток.
Затем добавь пустую строку и напиши:

📌 Краткое содержание:
[2-4 предложения, кратко описывающих основные темы и ключевые моменты разговора]"""

USER_PROMPT_TEMPLATE = """Отформатируй эту расшифровку. Язык аудио: {language}.

ТЕКСТ:
{text}"""


@dataclass
class FormattedResult:
    """Result of LLM formatting."""
    formatted_text: str
    model_used: str
    error: str = ""


def _sync_call_openrouter(api_key: str, model: str, text: str, language: str) -> str:
    """Call OpenRouter API synchronously (run in thread).
    
    Returns the formatted text from the LLM.
    Raises RuntimeError on failure.
    """
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "x-title": "VoiceToText Bot",
    }

    user_prompt = USER_PROMPT_TEMPLATE.format(
        language=language or "не определён",
        text=text,
    )

    # Merge system + user prompt into one message for maximum compatibility
    # (some free models like gemma-3-12b don't support system role)
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0.1,  # Low temperature for faithful formatting
        "max_tokens": 16000,
    }

    timeout = httpx.Timeout(timeout=300.0, connect=15.0, read=300.0, write=30.0)

    t0 = time.monotonic()
    with httpx.Client(timeout=timeout, http2=True) as client:
        resp = client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers=headers,
            json=payload,
        )

    elapsed = time.monotonic() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter {model} failed ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()

    # Check for API-level errors
    if "error" in data:
        raise RuntimeError(f"OpenRouter {model} error: {data['error']}")

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenRouter {model}: empty response")

    result = choices[0]["message"]["content"].strip()
    
    # Validate: since LLM must add punctuation AND a summary, the result 
    # must be longer than or equal to the input text. If shorter, it truncated.
    if len(result) < len(text):
        raise RuntimeError(
            f"LLM truncated text: {len(text)} -> {len(result)} chars. "
            f"Result must be >= original length."
        )
    
    logger.info("LLM formatting done with %s in %.1fs (%d -> %d chars)",
                model, elapsed, len(text), len(result))
    return result


# Cache for discovered free models
_cached_free_models: list[str] | None = None


def _sync_discover_free_models() -> list[str]:
    """Query OpenRouter API for available free models.
    
    Returns a list of free model IDs sorted by context window size (largest first).
    Only includes models with context >= 16000 (enough for long transcriptions).
    """
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0), http2=True) as client:
            resp = client.get(f"{OPENROUTER_BASE}/models")
        
        if resp.status_code != 200:
            logger.warning("Failed to fetch models from OpenRouter: %d", resp.status_code)
            return []

        data = resp.json()
        free_models = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            if ":free" in mid:
                ctx = m.get("context_length", 0)
                if ctx >= 16000:
                    free_models.append((mid, ctx))
        
        # Sort by context window (largest first = best for long texts)
        free_models.sort(key=lambda x: -x[1])
        result = [mid for mid, _ in free_models]
        logger.info("Discovered %d free models from OpenRouter (ctx>=16K)", len(result))
        return result
    except Exception as e:
        logger.warning("Could not fetch OpenRouter models: %s", e)
        return []


async def _get_models() -> list[str]:
    """Get the list of models to try, with dynamic discovery + configured fallback."""
    global _cached_free_models
    
    config = load_config()
    configured_models = config.openrouter.models
    
    # Try to discover available models (once, cached)
    if _cached_free_models is None:
        discovered = await asyncio.to_thread(_sync_discover_free_models)
        if discovered:
            _cached_free_models = discovered
            logger.info("Using %d discovered free models", len(discovered))
        else:
            _cached_free_models = []
            logger.info("Using configured model list (discovery failed)")
    
    # Merge: configured first (user preference), then discovered as fallback
    seen = set()
    merged = []
    for m in configured_models + _cached_free_models:
        if m not in seen:
            seen.add(m)
            merged.append(m)
    
    return merged


async def format_with_llm(text: str, language: str = "") -> FormattedResult:
    """Format transcription text using OpenRouter LLM.

    Tries each model in the fallback chain until one succeeds.
    On first call, discovers available free models from OpenRouter API.
    If no API key is configured, returns the original text unchanged.

    Args:
        text: Raw transcription text.
        language: Detected language code (e.g. "ru", "en").

    Returns:
        FormattedResult with formatted text and model info.
    """
    config = load_config()
    api_key = config.openrouter.api_key

    # If no API key — return text as-is (graceful degradation)
    if not api_key:
        logger.info("OpenRouter API key not set, skipping LLM formatting")
        return FormattedResult(formatted_text=text, model_used="none", error="API key not configured")

    if not text or len(text.strip()) < 10:
        return FormattedResult(formatted_text=text, model_used="none")

    # Get models: configured + dynamically discovered
    models = await _get_models()

    # Try each model in the fallback chain
    last_error = ""
    for model in models:
        try:
            logger.info("Trying LLM model: %s", model)
            formatted = await asyncio.to_thread(
                _sync_call_openrouter, api_key, model, text, language
            )
            return FormattedResult(formatted_text=formatted, model_used=model)
        except Exception as e:
            last_error = str(e)
            logger.warning("Model %s failed: %s", model, last_error)
            continue

    # All models failed — return original text
    logger.error("All %d LLM models failed. Last error: %s", len(models), last_error)
    return FormattedResult(
        formatted_text=text,
        model_used="none",
        error=f"Все модели недоступны: {last_error[:100]}",
    )

