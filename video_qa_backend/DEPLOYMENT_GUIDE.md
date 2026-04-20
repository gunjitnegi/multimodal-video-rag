# Deployment Guide - Multimodal Video QA System

This guide explains how to properly deploy the system for production or sharing.

## Method 1: Docker (Recommended)
Docker ensures that all dependencies (FFmpeg, Python, Node, Ollama) work perfectly across any machine.

### 1. Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- (Optional) Install NVIDIA Container Toolkit if you have a GPU.

### 2. Launching the System
Open a terminal in the root folder (`c:\final_year`) and run:
```bash
docker-compose up --build
```

### 3. Initialize AI Models
Once the containers are running, you need to tell the `ollama` container to download the models:
```bash
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull bakllava
```

The system will now be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000

---

## Method 2: Manual Deployment (VPS / Linux)

### 1. Install System Dependencies
On a clean Ubuntu server:
```bash
sudo apt update && sudo apt install -y ffmpeg python3-pip python3-venv git libgl1-mesa-glx
```

### 2. Setup Ollama
Install Ollama and pull models:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama pull bakllava
```

### 3. Backend Setup
```bash
cd video_qa_backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Frontend Setup
```bash
cd video_qa_frontend
npm install
npm run build
# Serve the 'dist' folder using Nginx
```

---

## Production Best Practices
1. **GPU Support**: Local LLMs are slow on CPU. Ensure your server has at least an NVIDIA RTX 3060 (12GB VRAM) for smooth performance.
2. **Reverse Proxy**: Use Nginx to handle SSL (HTTPS) and route traffic to the backend/frontend.
3. **Environment Variables**: Create a `.env` file in the backend to store API keys if you switch to Cloud models.
