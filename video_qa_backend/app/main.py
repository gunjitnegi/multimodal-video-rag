from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.core.database import engine, Base
from app.utils.helpers import setup_logging

from app.routers.video import router as video_router
from app.routers.chat import router as chat_router

app = FastAPI(
    title="Multimodal Video QA System",
    description="YouTube Video → Persistent Searchable Knowledge Base (2026)",
    version="0.4.0-phase3",
    debug=True
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)
app.include_router(chat_router)

@app.get("/")
async def root():
    return {
        "message": "Multimodal Video QA Backend - Phase 3 Active",
        "status": "healthy",
        "phase": "3 - Vector Search & Chat",
        "year": "2026",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.APP_ENV}

@app.on_event("startup")
async def startup_event():
    logger = setup_logging()
    logger.info("Starting Multimodal Video QA System - Phase 3")
    
    settings.create_directories()
    Base.metadata.create_all(bind=engine)
    
    logger.info("Database tables ready")
    logger.info(f"Data root: {settings.DATA_DIR.absolute()}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)