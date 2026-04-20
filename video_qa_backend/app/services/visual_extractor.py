import os
import cv2
import base64
import requests
import logging
from yt_dlp import YoutubeDL
from app.core.config import settings

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_VISION_MODEL = "bakllava"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "videos")
os.makedirs(DATA_DIR, exist_ok=True)

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def process_video_visuals(video_id: str):
    """
    Downloads a video, extracts frames every 10 seconds, and generates textual
    captions using LLaVA. Returns a list of visual chunks to be indexed.
    """
    video_path = os.path.join(DATA_DIR, f"{video_id}.mp4")
    visual_chunks = []

    # 1. Download Video
    logger.info(f"Downloading video {video_id} for visual extraction...")
    base_ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]', # Upgraded to 720p for better detail/OCR
        'outtmpl': video_path,
        'quiet': False,
        'no_warnings': True,
    }

    try:
        if not os.path.exists(video_path):
            download_success = False
            browsers_to_try = ['chrome', 'edge', 'firefox', 'brave', 'opera', None]
            
            for browser in browsers_to_try:
                ydl_opts = base_ydl_opts.copy()
                if browser:
                    ydl_opts['cookiesfrombrowser'] = (browser,)
                
                try:
                    logger.info(f"Attempting visual download with cookies from: {browser or 'None'}")
                    with YoutubeDL(ydl_opts) as ydl:
                        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                    download_success = True
                    logger.info("Video download finished.")
                    break
                except Exception as e:
                    logger.warning(f"Visual download failed with {browser or 'None'}: {e}")
                    
            if not download_success:
                logger.error(f"All visual download attempts failed for video {video_id}")
                return []
        else:
            logger.info("Video already downloaded.")
    except Exception as e:
        logger.error(f"Unexpected error during video download {video_id}: {e}")
        return []

    # 2. Extract Frames
    logger.info(f"Extracting frames from {video_id}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None or fps != fps:
        fps = 30.0 # fallback

    frame_interval = int(fps * 2) # Extract 1 frame every 2 seconds (was 5s) for better temporal coverage
    
    frame_count = 0
    extracted_frames = []
    
    # Create a specific folder for this video's frames if we are saving them
    video_frames_dir = None
    if settings.SAVE_VISUAL_FRAMES:
        video_frames_dir = settings.FRAMES_DIR / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Optionally save physical file
            if settings.SAVE_VISUAL_FRAMES and video_frames_dir:
                frame_path = video_frames_dir / f"frame_{int(frame_count / fps)}s.jpg"
                cv2.imwrite(str(frame_path), frame)

            # Encode frame to jpeg bytes
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                timestamp = frame_count / fps
                extracted_frames.append({
                    "timestamp": timestamp,
                    "bytes": buffer.tobytes()
                })
        
        frame_count += 1

    cap.release()
    total_frames = len(extracted_frames)
    logger.info(f"Extracted {total_frames} frames. Starting LLaVA vision analysis...")

    # 3. Caption frames using LLaVA
    for i, frame_data in enumerate(extracted_frames):
        ts = frame_data["timestamp"]
        b64_image = encode_image(frame_data["bytes"])

        logger.info(f"Analyzing frame {i+1}/{total_frames} (Timestamp: {ts:.1f}s)...")

        prompt = """Analyze this image EXHAUSTIVELY. DO NOT HALLUCINATE OR GUESS. If something is blurry or unclear, state "unclear".
1. SCENE & ENVIRONMENT: Describe the setting comprehensively. If indoors, describe the room type (e.g., house, kitchen, office) and key objects present (e.g., dining table, plates, sofa, computers). If outdoors, describe the environment (e.g., street, park, trees, buildings) and the time of day (e.g., day time, night time, sunset).
2. TEXT & CODE: Transcribe any visible text, headings, or code EXACTLY as written.
3. UI/DESIGN: Describe the layout and elements of any software/websites shown.
4. PEOPLE: Describe EVERY person individually. CLEARLY separate different individuals. 
- Distinguish between older people, middle-aged adults, and children.
- For EACH distinct person, explicitly state their precise clothing (color, type, accessories like hats/glasses). 
- DO NOT mix up the clothing or actions of different characters (e.g. do not say the old man is wearing what the middle-aged man is wearing).
Be highly detailed but strictly factual."""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_VISION_MODEL,
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False,
                    "options": {
                        "temperature": 0.0
                    }
                },
                timeout=120 # Vision models take longer
            )
            response.raise_for_status()
            
            caption = response.json().get("response", "").strip()
            
            if caption:
                visual_chunks.append({
                    "start": ts,
                    "end": ts + 10.0,
                    "text": f"[VISUAL] {caption}",
                    "type": "visual"
                })
                logger.info(f"Successfully generated caption for frame {i+1}.")

        except Exception as e:
            logger.warning(f"Failed to generate caption for frame {i+1} at {ts:.1f}s: {e}")

    logger.info(f"Visual extraction complete for {video_id}: generated {len(visual_chunks)} visual chunks.")
    return visual_chunks
