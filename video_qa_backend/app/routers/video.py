from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
import yt_dlp
import whisper
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

from app.core.database import get_db
from app.models.video import Video
from app.core.config import settings
from app.utils.helpers import get_video_paths
from app.services.transcript_chunker import chunk_transcript
from app.services.embedding_service import embedding_service
from app.services.visual_extractor import process_video_visuals

def process_and_index_transcript(youtube_id: str, segments: list, video_title: str):
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.youtube_id == youtube_id).first()
        if video:
            video.status = "Chunking Transcript..."
            db.commit()

        logger.info(f"Starting background chunking and indexing for {youtube_id}")
        chunks = chunk_transcript(segments)
        for chunk in chunks:
            chunk["type"] = "audio"
        
        if video:
            video.status = "Extracting Visual Frames (This may take a while)..."
            db.commit()

        # Process visual frames in parallel (runs synchronously here)
        visual_chunks = process_video_visuals(youtube_id)
        
        all_chunks = chunks + visual_chunks
        
        if video:
            video.status = "Saving Embeddings..."
            db.commit()

        embedding_service.save_index(youtube_id, all_chunks, video_title)
        
        if video:
            video.processed = True
            video.status = "Completed"
            db.commit()
            
        logger.info(f"Successfully chunked and indexed {youtube_id} ({len(chunks)} text, {len(visual_chunks)} visual)")
    except Exception as e:
        if video:
            video.status = f"Error: {str(e)}"
            db.commit()
        logger.error(f"Failed to chunk and index {youtube_id}: {str(e)}")
    finally:
        db.close()

router = APIRouter(prefix="/videos", tags=["videos"])
logger = logging.getLogger(__name__)

class VideoIngestRequest(BaseModel):
    youtube_url: str

class VideoResponse(BaseModel):
    id: int
    youtube_id: str
    title: str
    url: str
    thumbnail: Optional[str] = None
    duration: Optional[int] = None
    author: Optional[str] = None
    processed: bool = False
    created_at: str

class TranscriptResponse(BaseModel):
    youtube_id: str
    title: str
    transcript_path: str
    transcript: list
    text: str
    source: str   # "youtube_captions" or "whisper"

class VideoListResponse(BaseModel):
    videos: List[VideoResponse]

def extract_youtube_id(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("/")[0]
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    raise ValueError("Invalid YouTube URL format")

def format_datetime(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    return dt.isoformat()

# ===================== PHASE 1 - Ingestion =====================

import shutil
import os

@router.get("", response_model=VideoListResponse)
async def list_videos(db: Session = Depends(get_db)):
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    response_videos = []
    for v in videos:
        v_dict = {**v.__dict__}
        v_dict["created_at"] = format_datetime(v.created_at)
        response_videos.append(v_dict)
    return {"videos": response_videos}

@router.delete("/{youtube_id}")
async def delete_video(youtube_id: str, db: Session = Depends(get_db)):
    try:
        video = db.query(Video).filter(Video.youtube_id == youtube_id).first()
        if video:
            db.delete(video)
            db.commit()

        # Delete physical files
        files_to_remove = [
            settings.VIDEOS_DIR / f"{youtube_id}.mp4",
            settings.AUDIO_DIR / f"{youtube_id}.wav",
            settings.AUDIO_DIR / f"{youtube_id}.m4a",
            settings.AUDIO_DIR / f"{youtube_id}.webm",
            settings.AUDIO_DIR / f"{youtube_id}.mp4",
            settings.TRANSCRIPTS_DIR / f"{youtube_id}.json",
            settings.TRANSCRIPTS_DIR / f"{youtube_id}.txt",
        ]
        for f in files_to_remove:
            try:
                if f.exists():
                    os.remove(f)
            except Exception as e:
                logger.warning(f"Failed to delete file {f}: {e}")

        # Delete directories
        dirs_to_remove = [
            settings.EMBEDDINGS_DIR / youtube_id,
            settings.FRAMES_DIR / youtube_id
        ]
        for d in dirs_to_remove:
            try:
                if d.exists():
                    shutil.rmtree(d)
            except Exception as e:
                logger.warning(f"Failed to delete directory {d}: {e}")

        logger.info(f"Completely deleted video {youtube_id} and all associated files.")
        return {"status": "success", "message": f"Deleted {youtube_id}"}
    except Exception as e:
        logger.error(f"Deletion error for {youtube_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=VideoResponse)
async def ingest_video(
    request: VideoIngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        youtube_id = extract_youtube_id(request.youtube_url)

        existing = db.query(Video).filter(Video.youtube_id == youtube_id).first()
        if existing:
            logger.info(f"Video already exists: {youtube_id}")
            if not existing.processed:
                background_tasks.add_task(process_audio_background, youtube_id, db)
            return {**existing.__dict__, "created_at": format_datetime(existing.created_at)}

        ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.youtube_url, download=False)

        video = Video(
            youtube_id=youtube_id,
            title=info.get('title', 'Unknown Title'),
            url=request.youtube_url,
            thumbnail=info.get('thumbnail'),
            duration=info.get('duration'),
            author=info.get('uploader'),
            description=info.get('description'),
            processed=False
        )

        db.add(video)
        db.commit()
        db.refresh(video)

        background_tasks.add_task(process_audio_background, youtube_id, db)

        logger.info(f"Video metadata saved. Background audio download started: {youtube_id}")

        return {**video.__dict__, "created_at": format_datetime(video.created_at)}

    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def process_audio_background(youtube_id: str, db: Session):
    """Download + FORCE clean .wav conversion"""
    try:
        video = db.query(Video).filter(Video.youtube_id == youtube_id).first()
        if not video:
            return

        paths = get_video_paths(youtube_id)
        ffmpeg_exe = str(Path(settings.BASE_DIR) / "venv" / "Scripts" / "ffmpeg.exe")

        base_ydl_opts = {
            'outtmpl': str(settings.AUDIO_DIR / f"{youtube_id}.%(ext)s"),
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'ffmpeg_location': str(Path(ffmpeg_exe).parent),
            'quiet': False,
            'retries': 5,
        }

        download_success = False
        last_error = None
        browsers_to_try = ['chrome', 'edge', 'firefox', 'brave', 'opera', None]
        
        for browser in browsers_to_try:
            ydl_opts = base_ydl_opts.copy()
            if browser:
                ydl_opts['cookiesfrombrowser'] = (browser,)
            
            try:
                logger.info(f"Attempting download with cookies from: {browser or 'None'}")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
                download_success = True
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Download failed with {browser or 'None'}: {e}")
                
        if not download_success:
            raise Exception(f"All audio download attempts failed. Last error: {last_error}")

        # Find downloaded file
        downloaded = None
        for ext in ['.m4a', '.mp4']:
            candidate = settings.AUDIO_DIR / f"{youtube_id}{ext}"
            if candidate.exists():
                downloaded = candidate
                break

        if not downloaded:
            raise Exception("Downloaded audio file not found")

        # Convert to clean wav
        wav_path = paths["audio"]
        cmd = [
            ffmpeg_exe,
            "-i", str(downloaded),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(wav_path),
            "-y"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr}")
            raise Exception("ffmpeg conversion failed")

        video.audio_path = str(wav_path)
        video.status = "Audio Ready. Awaiting Processing..."
        db.commit()

        logger.info(f"Clean .wav ready: {video.audio_path}")

    except Exception as e:
        logger.error(f"Audio processing failed for {youtube_id}: {str(e)}")
        if video:
            video.processed = False
            db.commit()

# ===================== PHASE 2 - Transcript =====================
@router.post("/{youtube_id}/transcribe", response_model=TranscriptResponse)
async def transcribe_video(
    youtube_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        video = db.query(Video).filter(Video.youtube_id == youtube_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        transcript_path = settings.TRANSCRIPTS_DIR / f"{youtube_id}.json"

        # PRIMARY: YouTube Transcript API (Manual or Auto-generated)
        try:
            logger.info(f"Trying YouTube Transcript API for {youtube_id}")

            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.list(youtube_id)

            try:
                transcript = transcript_list.find_manually_created_transcript(['hi', 'en-IN', 'en'])
                logger.info("Using manually created YouTube captions")
            except Exception:
                transcript = transcript_list.find_generated_transcript(['hi', 'en-IN', 'en'])
                logger.info("Using auto-generated YouTube captions")

            transcript_data = transcript.fetch()

            text = " ".join([item.text for item in transcript_data])
            segments = [{"start": item.start,"duration": getattr(item, "duration", 0),"text": item.text} for item in transcript_data]
            transcript_save = {
                "text": text,
                "segments": segments,
                "language": transcript.language_code
            }

            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_save, f, ensure_ascii=False, indent=2)

            with open(settings.TRANSCRIPTS_DIR / f"{youtube_id}.txt", 'w', encoding='utf-8') as f:
                f.write(text)

            video.transcript_path = str(transcript_path)
            video.processed = True
            db.commit()

            background_tasks.add_task(process_and_index_transcript, youtube_id, segments, video.title)

            logger.info(f" YouTube captions used successfully for {youtube_id}")
            return {
                "youtube_id": youtube_id,
                "title": video.title,
                "transcript_path": str(transcript_path),
                "transcript": segments,
                "text": text,
                "source": "youtube_captions"
            }

        except (NoTranscriptFound, TranscriptsDisabled):
            logger.info("No YouTube captions available → falling back to Whisper small model")
        except Exception as e:
            logger.warning(f"YouTube captions failed: {str(e)} → falling back to Whisper")

        # FALLBACK: Whisper "small" model (best practical accuracy on CPU)
        if not video.audio_path or not Path(video.audio_path).exists():
            raise HTTPException(status_code=400, detail="Clean audio (.wav) not found. Re-ingest the video.")

        logger.info(f"Starting Whisper small model fallback for {youtube_id} ...")
        model = whisper.load_model("small", device=settings.get_device)

        result = model.transcribe(str(video.audio_path), fp16=False, language=None)

        transcript_data = {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"]
        }

        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

        with open(settings.TRANSCRIPTS_DIR / f"{youtube_id}.txt", 'w', encoding='utf-8') as f:
            f.write(result["text"])

        video.transcript_path = str(transcript_path)
        video.processed = True
        db.commit()

        background_tasks.add_task(process_and_index_transcript, youtube_id, result["segments"], video.title)

        logger.info("Transcription completed using Whisper small model")

        return {
            "youtube_id": youtube_id,
            "title": video.title,
            "transcript_path": str(transcript_path),
            "transcript": result["segments"],
            "text": result["text"],
            "source": "whisper_small"
        }

    except Exception as e:
        logger.error(f"Transcription failed for {youtube_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=VideoListResponse)
async def list_videos(db: Session = Depends(get_db)):
    videos = db.query(Video).all()
    return {"videos": [{**v.__dict__, "created_at": format_datetime(v.created_at)} for v in videos]}

@router.get("/{youtube_id}", response_model=VideoResponse)
async def get_video(youtube_id: str, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.youtube_id == youtube_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return {**video.__dict__, "created_at": format_datetime(video.created_at)}