import os
import logging
from pathlib import Path
from app.core.config import settings

def setup_logging():
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.LOGS_DIR / "app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_video_paths(video_id: str):
    """Return consistent paths for a video"""
    return {
        "video": settings.VIDEOS_DIR / f"{video_id}.mp4",
        "audio": settings.AUDIO_DIR / f"{video_id}.wav",
        "transcript": settings.TRANSCRIPTS_DIR / f"{video_id}.json",
        "faiss_index": settings.EMBEDDINGS_DIR / f"{video_id}.faiss"
    }