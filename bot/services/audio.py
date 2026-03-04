"""Audio extraction and conversion service using FFmpeg."""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported input formats
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus", ".oga"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".3gp"}
ALL_MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Temporary directory for processing
TEMP_DIR = Path(tempfile.gettempdir()) / "voicetotext"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def is_supported_format(filename: str) -> bool:
    """Check if file extension is supported."""
    ext = Path(filename).suffix.lower()
    return ext in ALL_MEDIA_EXTENSIONS


def get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


async def extract_audio(input_path: str, output_path: str = None) -> str:
    """Extract audio from media file and convert to WAV 16kHz mono.

    This format is optimal for speech recognition APIs.

    Args:
        input_path: Path to input media file.
        output_path: Optional path for output WAV file.
                     If not provided, generates a temp path.

    Returns:
        Path to the extracted WAV file.

    Raises:
        RuntimeError: If FFmpeg conversion fails.
    """
    if output_path is None:
        # Generate output path in temp directory
        stem = Path(input_path).stem
        output_path = str(TEMP_DIR / f"{stem}_{os.getpid()}.wav")

    cmd = [
        "ffmpeg",
        "-loglevel", "error", # suppress warning spam
        "-i", input_path,
        "-vn",              # no video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",     # 16kHz sample rate (optimal for STT)
        "-ac", "1",         # mono
        "-y",               # overwrite output
        output_path,
    ]

    logger.info("Extracting audio: %s -> %s", input_path, output_path)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode().strip()
        logger.error("FFmpeg error: %s", error_msg)
        raise RuntimeError(f"FFmpeg conversion failed: {error_msg[:200]}")

    logger.info("Audio extracted successfully: %s", output_path)
    return output_path


async def get_audio_duration(file_path: str) -> float:
    """Get duration of audio/video file in seconds using ffprobe.

    Returns:
        Duration in seconds, or 0.0 if unable to determine.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            duration = float(stdout.decode().strip())
            logger.info("File duration: %.1f seconds", duration)
            return duration
    except (ValueError, OSError) as e:
        logger.warning("Could not determine file duration: %s", e)

    return 0.0


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string (MM:SS or HH:MM:SS)."""
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def cleanup_temp_file(file_path: str) -> None:
    """Remove temporary file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug("Cleaned up temp file: %s", file_path)
    except OSError as e:
        logger.warning("Failed to clean up temp file %s: %s", file_path, e)
