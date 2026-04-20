from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import logging
import requests

from app.core.database import get_db
from app.models.video import Video
from app.core.config import settings
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/videos", tags=["chat"])


# -------------------------
# OLLAMA CONFIG
# -------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"


# -------------------------
# REQUEST / RESPONSE
# -------------------------
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    session_id: Optional[str] = None
    search_mode: Optional[str] = "both"  # "audio", "visual", or "both"


class ChatResponse(BaseModel):
    youtube_id: str
    query: str
    relevant_chunks: List[dict]
    answer: str


# -------------------------
# MEMORY
# -------------------------
CHAT_MEMORY = {}


def get_chat_history(session_id: str):
    return CHAT_MEMORY.get(session_id, [])


def update_chat_history(session_id: str, user_q: str, answer: str):
    if session_id not in CHAT_MEMORY:
        CHAT_MEMORY[session_id] = []

    CHAT_MEMORY[session_id].append({
        "user": user_q,
        "assistant": answer
    })

    CHAT_MEMORY[session_id] = CHAT_MEMORY[session_id][-5:]


# -------------------------
# OLLAMA CALL
# -------------------------
def call_ollama(prompt: str, temperature: float = 0.2) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=30
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return ""


# -------------------------
# QUERY REWRITE
# -------------------------
def rewrite_query(query: str, history: List[dict]) -> str:
    if not history:
        return query

    prompt = f"""
Rewrite the query clearly and standalone.

History:
{history}

Query:
{query}

Rewritten:
"""
    result = call_ollama(prompt, temperature=0.0)
    return result if result else query


# -------------------------
# 🔥 BUILD CONTEXT (FIXED)
# -------------------------
def build_context(chunks: List[dict]) -> str:
    context_blocks = []

    for i, chunk in enumerate(chunks):
        block = f"""
[Chunk {i+1}]
Time: {chunk.get('start', 0)}s - {chunk.get('end', 0)}s
Summary: {chunk.get('summary', '')}

Content:
{chunk['text']}
"""
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


# -------------------------
# 🔥 DETECT VIDEO TYPE
# -------------------------
def detect_video_type(title: str, description: str) -> str:
    combined = f"{title} {description}".lower()
    if any(k in combined for k in ["official video", "music video", "song", "lyrics", "prod. by", "singer", "rapper", "seedhe maut"]):
        return "song"
    return "general"


# -------------------------
# 🔥 STRONG ANSWER GENERATION
# -------------------------
def generate_answer(query: str, context: str, history: List[dict], video_title: str, video_desc: str = "", video_type: str = "general") -> str:
    if video_type == "song":
        rules = """
- This is a music video/song. The context contains rap/song lyrics in Hindi/Hinglish.
- CRITICAL: Identify the artist from the Video Title and Description. Use your WORLD KNOWLEDGE about the artist to enrich the answer.
- Refer to the artists by their actual name, do NOT say "the individuals".
- Mentally translate the lyrics into English to understand the deep meaning.
- Explain the profound meaning, struggle, themes, and authenticity behind the lyrics.
- Write a profound, well-structured thematic analysis using bullet points (e.g., Authenticity, Resilience).
- Do NOT hallucinate literal political events from metaphors.
"""
    else:
        rules = """
- You may use your WORLD KNOWLEDGE to map character names or entities in the Question (like 'Jethalal', 'Tapu') to the generic descriptions in the [VISUAL] chunks (like 'a man', 'a boy').
- CRITICAL: Other than identifying people by name, you MUST answer ONLY using the context chunks provided.
- CRITICAL ANTI-HALLUCINATION RULE: NEVER guess or hallucinate clothing, food, code snippets, or UI layouts. If the visual context does not explicitly describe exactly what the person is wearing, eating, or what code is on the screen, you MUST say "The visual context does not show this."
- Do NOT write "Chunk 1 says..." or summarize chunk by chunk. Synthesize the information natively.
- You MUST format your answer EXACTLY like this:

[Brief 1-2 sentence overview of the topic]

**Key Takeaways:**
* **[Key Concept 1]:** [Detailed explanation]
* **[Key Concept 2]:** [Detailed explanation]

- Extract the core concepts, visual details, and examples mentioned in the video.
- Ignore noise or filler, but provide comprehensive explanations.
"""

    prompt = f"""
You are an advanced video analysis AI.

Video Title: {video_title}
Video Description: {video_desc}

Context:
{context}

Question:
{query}

CRITICAL INSTRUCTIONS FOR YOUR ANSWER:
{rules.strip()}
- Some context chunks begin with [VISUAL]. These are textual descriptions of the video frames. Use them to answer visual questions.
- You MUST cite your sources using chunk IDs like [1], [2].

Answer:
"""

    response = call_ollama(prompt, temperature=0.2)

    return response if response else ""


# -------------------------
# 🔥 FALLBACK ANSWER (CRITICAL)
# -------------------------
def fallback_answer(query: str, chunks: List[dict]) -> str:
    # fallback = return best chunk summary
    best = chunks[0]

    return f"""
Likely answer (from best matching segment):

{best.get('summary', best['text'])}

Time: {best.get('start')}s - {best.get('end')}s
"""


# -------------------------
# MAIN ENDPOINT
# -------------------------
@router.post("/{youtube_id}/chat", response_model=ChatResponse)
async def chat_with_video(
    youtube_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    try:
        video = db.query(Video).filter(Video.youtube_id == youtube_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        index_dir = Path(settings.EMBEDDINGS_DIR) / youtube_id
        if not (index_dir / "index.faiss").exists():
            raise HTTPException(
                status_code=400,
                detail="Embeddings not found. Process video first."
            )

        history = get_chat_history(request.session_id) if request.session_id else []

        rewritten_query = rewrite_query(request.query, history)

        relevant_chunks = embedding_service.search(
            youtube_id,
            rewritten_query,
            request.top_k,
            search_mode=request.search_mode
        )

        if not relevant_chunks:
            return {
                "youtube_id": youtube_id,
                "query": request.query,
                "relevant_chunks": [],
                "answer": "No relevant information found in the video."
            }

        # 🔥 SORT BY SCORE (IMPORTANT)
        relevant_chunks = sorted(relevant_chunks, key=lambda x: x["score"], reverse=True)

        context = build_context(relevant_chunks)

        video_type = detect_video_type(video.title, video.description or "")

        answer = generate_answer(
            query=rewritten_query,
            context=context,
            history=history,
            video_title=video.title,
            video_desc=video.description or "",
            video_type=video_type
        )

        # 🔥 FALLBACK IF MODEL FAILS
        if not answer or "not found" in answer.lower():
            answer = fallback_answer(request.query, relevant_chunks)

        if request.session_id:
            update_chat_history(request.session_id, request.query, answer)

        return {
            "youtube_id": youtube_id,
            "query": request.query,
            "relevant_chunks": relevant_chunks,
            "answer": answer
        }

    except Exception as e:
        logger.error(f"Chat failed for {youtube_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))