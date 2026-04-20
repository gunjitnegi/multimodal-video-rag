# Multimodal Video QA System

An advanced, end-to-end Video Question Answering (QA) system that allows users to ingest YouTube videos and interact with them using an AI that "sees" and "hears" the content.

![Project Banner](https://img.shields.io/badge/AI-Multimodal-blueviolet?style=for-the-badge)
![Tech](https://img.shields.io/badge/FastAPI-React-blue?style=for-the-badge)
![Vision](https://img.shields.io/badge/BakLLaVA-Llama3.1-orange?style=for-the-badge)

## 🌟 Key Features

- **Multimodal Intelligence**: Answers questions about both spoken content (audio) and visual elements (wearing, actions, environment).
- **Robust Ingestion**: Integrated `yt-dlp` with multi-browser cookie support to bypass bot detection.
- **High-Fidelity Vision**: Uses **BakLLaVA** to analyze 720p frames every 2 seconds.
- **Hybrid Search**: Combines **FAISS** (Semantic) and **BM25** (Keyword) for precise retrieval.
- **Source Tracing**: Provides timestamps and citations for every answer.
- **Anti-Hallucination**: Zero-temperature prompting and strict constraints ensure factual accuracy.

## 🛠️ Tech Stack

### Backend & AI
- **Framework**: FastAPI (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **LLM Orchestration**: Ollama
- **Models**: Llama 3.1 8B (Reasoning), BakLLaVA (Vision), Whisper (Transcription)
- **Embeddings**: E5-base via Sentence-Transformers
- **Vector DB**: FAISS

### Frontend
- **Framework**: React.js + Vite
- **UI/UX**: Modern Glassmorphism design with Lucide Icons

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js & npm
- [FFmpeg](https://ffmpeg.org/download.html) installed in system PATH
- [Ollama](https://ollama.com/) installed and running

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gunjitnegi/multimodal-video-rag.git
   cd multimodal-video-rag
   ```

2. **Backend Setup**:
   ```bash
   cd video_qa_backend
   pip install -r requirements.txt
   ollama pull llama3.1:8b
   ollama pull bakllava
   python -m uvicorn app.main:app --reload
   ```

3. **Frontend Setup**:
   ```bash
   cd ../video_qa_frontend
   npm install
   npm run dev
   ```

## 🐳 Docker Deployment

The project includes a `docker-compose.yml` for simplified deployment:
```bash
docker-compose up --build
```

## 📄 License
Final Year Project - 2026
