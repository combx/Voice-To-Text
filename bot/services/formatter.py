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
1. НЕ УДАЛЯЙ ни одного слова из оригинального текста (если не запрошен перевод). Каждое слово должно остаться на месте.
2. НЕ ЗАМЕНЯЙ слова на синонимы. НЕ перефразируй.
3. НЕ ДОБАВЛЯЙ новых слов (кроме знаков препинания).
4. СОХРАНЯЙ все метки спикеров (🔵 Спикер 1 / 🔵 Speaker 1, 🟢 Спикер 2 / 🟢 Speaker 2 и т.д.) и временные метки [MM:SS] в точности как в оригинале.
5. РАЗРЕШЕНО только:
   - Расставить знаки препинания (запятые, точки, вопросительные и восклицательные знаки, тире, двоеточия)
   - Расставить заглавные буквы в начале предложений и для имён собственных
   - Разбить длинные реплики одного спикера на НЕБОЛЬШИЕ АБЗАЦЫ (по 2-3 предложения) для удобства чтения. Никаких огромных сплошных блоков текста!
6. ПИШИ ВЕСЬ ТЕКСТ (включая заголовок 'Краткое содержание' и само содержание) НА ТОМ ЖЕ ЯЗЫКЕ, ЧТО И ОРИГИНАЛЬНАЯ РАСШИФРОВКА (если не запрошен перевод).

ФОРМАТ ОТВЕТА:
Строго следуй переданной инструкции о том, куда поместить "Краткое содержание" (в начало или в конец текста).
Сам текст должен сохранять все метки спикеров и временные метки."""

USER_PROMPT_TEMPLATE = """Отформатируй эту расшифровку. Язык аудио: {language}.

{summary_instruction}

ТЕКСТ:
{text}"""


@dataclass
class FormattedResult:
    """Result of LLM formatting."""
    formatted_text: str
    model_used: str
    error: str = ""


def _sync_call_openrouter(
    api_key: str, 
    model: str, 
    text: str, 
    language: str, 
    duration_seconds: float,
    target_language: str | None = None
) -> str:
    """Call OpenRouter API synchronously (run in thread).
    
    Returns the formatted text from the LLM.
    Raises RuntimeError on failure.
    """
    from bot.services.transcriber import LOCALIZATIONS
    loc = LOCALIZATIONS.get(language, LOCALIZATIONS["en"]) if language in LOCALIZATIONS else LOCALIZATIONS["en"]
    summary_title = loc["summary_title"]

    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "x-title": "VoiceToText Bot",
    }

    if target_language == "ru":
        # Translation mode
        summary_instruction = (
            "ЗАДАЧА: Переведи весь текст выше на РУССКИЙ ЯЗЫК. Сохрани все метки спикеров и таймкоды. "
            "СНАЧАЛА напиши Краткое содержание на русском (заголовок '📌 Краткое содержание:'), "
            "затем пустую строку, а ПОТОМ переведенный текст расшифровки."
        )
    elif duration_seconds > 40:
        summary_instruction = (
            f"Так как аудио длинное, СНАЧАЛА напиши Краткое содержание (2-4 предложения "
            f"с заголовком '{summary_title}:'), затем пустую строку, "
            f"а ПОТОМ отформатированный текст. ПИШИ ВСЁ НА ЯЗЫКЕ ОРИГИНАЛА ({language})."
        )
    else:
        summary_instruction = (
            f"СНАЧАЛА выведи отформатированный текст, а В КОНЦЕ текста добавь пустую строку "
            f"и напиши Краткое содержание (2-4 предложения с заголовком '{summary_title}:'). "
            f"ПИШИ ВСЁ НА ЯЗЫКЕ ОРИГИНАЛА ({language})."
        )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        language=language or "не определён",
        summary_instruction=summary_instruction,
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

    # Уменьшенный таймаут для одной модели (было 300)
    timeout = httpx.Timeout(timeout=60.0, connect=15.0, read=60.0, write=30.0)

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
    # Validate length to prevent truncation bugs. 
    # For verbatim formatting, it should be >= original length (due to added summary & punctuation).
    # For translations, it can naturally be shorter, so we use a relaxed 50% threshold.
    if target_language:
        if len(result) < len(text) * 0.5:
            raise RuntimeError(
                f"LLM translation too short: {len(text)} -> {len(result)} chars. "
                "Output must be at least 50% of original length."
            )
    else:
        # Strict verbatim check
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

FAST_PAID_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
]

async def get_openrouter_balance(api_key: str) -> dict:
    """Fetch OpenRouter account balance via /v1/credits API.
    
    Returns a dict with 'total_credits', 'total_usage', 'remaining' in dollars.
    """
    if not api_key:
        return {"total_credits": 0.0, "total_usage": 0.0, "remaining": 0.0}
    
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{OPENROUTER_BASE}/credits"
    
    try:
        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            resp = await client.get(url, headers=headers)
            
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            total_credits = data.get("total_credits", 0.0)
            total_usage = data.get("total_usage", 0.0)
            remaining = max(0.0, total_credits - total_usage)
            return {
                "total_credits": total_credits,
                "total_usage": total_usage,
                "remaining": remaining,
            }
    except Exception as e:
        logger.warning("Could not fetch OpenRouter balance: %s", e)
        
    return {"total_credits": 0.0, "total_usage": 0.0, "remaining": 0.0}


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


async def _get_models(mode: str = "paid") -> list[str]:
    """Get the list of models to try, based on user mode."""
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
    
    if mode == "paid":
        logger.info("Using paid models with free fallback: %s", FAST_PAID_MODELS)
        for m in FAST_PAID_MODELS:
            if m not in seen:
                seen.add(m)
                merged.append(m)
                
    for m in configured_models + _cached_free_models:
        if m not in seen:
            seen.add(m)
            merged.append(m)
    
    return merged


async def _try_all_models(
    text: str,
    language: str,
    duration_seconds: float,
    target_language: str | None,
    models: list[str],
    api_key: str
) -> FormattedResult:
    """Try each model in sequence for formatting."""
    last_error = ""
    for model in models:
        try:
            logger.info("Trying LLM model: %s", model)
            formatted = await asyncio.to_thread(
                _sync_call_openrouter, api_key, model, text, language, duration_seconds, target_language
            )
            return FormattedResult(formatted_text=formatted, model_used=model)
        except Exception as e:
            last_error = str(e)
            logger.warning("Model %s failed: %s", model, last_error)
            continue

    logger.error("All %d LLM models failed. Last error: %s", len(models), last_error)
    return FormattedResult(
        formatted_text=text,
        model_used="none",
        error=f"Все модели недоступны: {last_error[:100]}",
    )


# Максимальное время на перебор всех моделей (секунды)
LLM_TOTAL_TIMEOUT = 90.0

async def format_with_llm(
    text: str,
    language: str = "",
    duration_seconds: float = 0.0,
    target_language: str | None = None,
    mode: str = "paid",
) -> FormattedResult:
    """Format transcription text using OpenRouter LLM.

    Tries each model in the fallback chain until one succeeds.
    Total execution is bounded by LLM_TOTAL_TIMEOUT.
    """
    config = load_config()
    api_key = config.openrouter.api_key

    if not api_key:
        logger.info("OpenRouter API key not set, skipping LLM formatting")
        return FormattedResult(formatted_text=text, model_used="none", error="API key not configured")

    if not text or len(text.strip()) < 10:
        return FormattedResult(formatted_text=text, model_used="none")

    models = await _get_models(mode)

    try:
        # Wrap the whole fallback chain in a single timeout
        result = await asyncio.wait_for(
            _try_all_models(text, language, duration_seconds, target_language, models, api_key),
            timeout=LLM_TOTAL_TIMEOUT
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(
            "LLM formatting timed out after %.0fs total. Returning raw text.",
            LLM_TOTAL_TIMEOUT
        )
        return FormattedResult(
            formatted_text=text,
            model_used="none",
            error="Timeout ожидания LLM — выведен оригинальный текст"
        )


