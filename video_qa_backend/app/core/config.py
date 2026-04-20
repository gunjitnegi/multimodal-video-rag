import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

class Settings:
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    
    VIDEOS_DIR: Path = DATA_DIR / "videos"
    AUDIO_DIR: Path = DATA_DIR / "audio"
    TRANSCRIPTS_DIR: Path = DATA_DIR / "transcripts"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    FRAMES_DIR: Path = DATA_DIR / "frames"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Added missing attribute
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Toggle to save physical .jpg files for debugging/frontend
    SAVE_VISUAL_FRAMES: bool = os.getenv("SAVE_VISUAL_FRAMES", "true").lower() == "true"
    
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    @property
    def get_device(self) -> str:
        # Force CPU because PyTorch 2.5.1 wheel does not support RTX 5050 (Blackwell sm_120) yet
        return "cpu"
    
    def create_directories(self):
        for directory in [self.VIDEOS_DIR, self.AUDIO_DIR, self.TRANSCRIPTS_DIR, 
                         self.EMBEDDINGS_DIR, self.FRAMES_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

settings = Settings()