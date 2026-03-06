"""YouTube service for extracting subtitles and metadata using yt-dlp."""

import asyncio
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class YouTubeMetadata:
    """Metadata for a YouTube video."""
    video_id: str
    title: str
    duration_seconds: int
    language: str | None = None
    subtitles_path: str | None = None
    subtitles_text: str | None = None

def _sync_extract_subtitles(url: str, temp_dir: str) -> dict:
    """Run yt-dlp to extract subtitles and metadata synchronously."""
    import yt_dlp
    
    # Options for yt-dlp
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['ru', 'en', '.*'], # Priority: RU, then EN, then any
        'outtmpl': f'{temp_dir}/%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        video_id = info.get('id')
        title = info.get('title')
        duration = info.get('duration')
        
        # Check for available subtitles
        subtitles = info.get('requested_subtitles')
        sub_text = None
        lang = None
        
        if subtitles:
            # Pick the best one (prefer RU if available, else whatever yt-dlp gave us)
            # Find the actual file yt-dlp would download or has downloaded
            # (since skip_download is True, it might not have saved them yet in extract_info)
            # We need to actually download just the subtitles.
            ydl.download([url])
            
            # Find the downloaded .vtt or .srt file
            for ext in ['ru.vtt', 'ru.srt', 'en.vtt', 'en.srt', 'vtt', 'srt']:
                potential_file = Path(temp_dir) / f"{video_id}.{ext}"
                if potential_file.exists():
                    sub_text = _clean_vtt(potential_file.read_text(encoding='utf-8'))
                    lang = ext.split('.')[0] if '.' in ext else 'unknown'
                    break
        
        return {
            'video_id': video_id,
            'title': title,
            'duration': duration,
            'language': lang,
            'subtitles_text': sub_text
        }

def _clean_vtt(vtt_text: str) -> str:
    """Clean VTT/SRT format to plain text."""
    import re
    # Remove WEBVTT header
    text = re.sub(r'WEBVTT', '', vtt_text)
    # Remove timestamps and metadata
    text = re.sub(r'\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3}.*?\n', '', text)
    # Remove HTML-like tags
    text = re.sub(r'<.*?>', '', text)
    # Remove line numbers (for SRT)
    text = re.sub(r'^\d+\n', '', text, flags=re.MULTILINE)
    # Remove duplicated lines (common in auto-subs)
    lines = text.split('\n')
    cleaned_lines = []
    prev_line = ""
    for line in lines:
        line = line.strip()
        if line and line != prev_line:
            cleaned_lines.append(line)
            prev_line = line
    
    return " ".join(cleaned_lines)

async def get_youtube_subs(url: str, temp_dir: str) -> YouTubeMetadata | None:
    """Extract subtitles and metadata from a YouTube URL."""
    try:
        data = await asyncio.to_thread(_sync_extract_subtitles, url, temp_dir)
        if not data.get('subtitles_text'):
            logger.warning("No subtitles found for YouTube URL: %s", url)
            return YouTubeMetadata(
                video_id=data['video_id'],
                title=data['title'],
                duration_seconds=data['duration']
            )
            
        return YouTubeMetadata(
            video_id=data['video_id'],
            title=data['title'],
            duration_seconds=data['duration'],
            language=data['language'],
            subtitles_text=data['subtitles_text']
        )
    except Exception as e:
        logger.error("Error extracting YouTube subtitles: %s", e)
        return None
