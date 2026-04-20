from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    youtube_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    url = Column(String, nullable=False)
    thumbnail = Column(String)
    duration = Column(Integer)
    author = Column(String)
    description = Column(Text)
    transcript_path = Column(String)
    audio_path = Column(String)
    faiss_index_path = Column(String)
    processed = Column(Boolean, default=False)
    status = Column(String, default="Pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Video(youtube_id={self.youtube_id}, title={self.title})>"