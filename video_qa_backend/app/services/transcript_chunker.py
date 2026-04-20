from typing import List, Dict
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import requests
import hashlib

from app.core.config import settings

logger = logging.getLogger(__name__)

_model = SentenceTransformer('all-MiniLM-L6-v2', device=settings.get_device)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct"

SUMMARY_CACHE = {}

# -------------------------
# CLEANING
# -------------------------
def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\.\.\.', '.', text)
    text = re.sub(r'(\buh\b|\bum\b|\byou know\b|\blike\b)\s*', '', text, flags=re.IGNORECASE)
    return text


# -------------------------
# 🚫 NOISE FILTER (CRITICAL)
# -------------------------
def _is_noise(text: str) -> bool:
    text = text.strip().lower()

    if len(text) < 15:
        return True
    if any(tag in text for tag in ["[music]", "[संगीत]", "♪"]):
        return True

    return False


# -------------------------
# TOKEN ESTIMATION
# -------------------------
def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.2)


# -------------------------
# DOMAIN DETECTION
# -------------------------
def _detect_domain(texts: List[str]) -> str:
    joined = " ".join(texts[:20]).lower()

    if any(k in joined for k in ["code", "function", "python", "algorithm"]):
        return "technical"
    if any(k in joined for k in ["interview", "podcast", "discussion"]):
        return "podcast"
    if any(k in joined for k in ["[music]", "♪", "chorus", "verse", "song"]):
        return "song"

    return "general"


# -------------------------
# CACHE KEY
# -------------------------
def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# -------------------------
# 🔥 BATCH LLM SUMMARY
# -------------------------
def _batch_summarize(chunks: List[Dict]) -> Dict:

    uncached = []
    results = {}

    for chunk in chunks:
        key = _hash_text(chunk["text"])

        if key in SUMMARY_CACHE:
            results[key] = SUMMARY_CACHE[key]
        else:
            uncached.append((key, chunk))

    if uncached:
        prompt = "Summarize each chunk clearly in 1-2 lines.\n\n"

        for i, (_, chunk) in enumerate(uncached):
            prompt += f"Chunk {i+1}:\n{chunk['text']}\n\n"

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=20
            )

            raw_output = response.json().get("response", "")
            outputs = [o.strip() for o in raw_output.split("\n") if o.strip()]

            for (key, chunk), summary in zip(uncached, outputs):
                data = {
                    "summary": summary,
                    "citations": [{"start": chunk["start"], "end": chunk["end"]}]
                }
                SUMMARY_CACHE[key] = data
                results[key] = data

        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")

        # Fallback for any chunks that were missed (either due to exception or LLM returning too few lines)
        for key, chunk in uncached:
            if key not in results:
                fallback = {
                    "summary": chunk["text"][:200],
                    "citations": [{"start": chunk["start"], "end": chunk["end"]}]
                }
                SUMMARY_CACHE[key] = fallback
                results[key] = fallback

    return results


# -------------------------
# SPEAKER
# -------------------------
def _get_speaker(seg: Dict):
    return seg.get("speaker", "unknown")


# -------------------------
# ADAPTIVE THRESHOLD
# -------------------------
def _adaptive_threshold(similarities: List[float]) -> float:
    if not similarities:
        return 0.6
    return max(0.5, np.mean(similarities) - np.std(similarities))


# -------------------------
# 🚀 MAIN CHUNKER
# -------------------------
def chunk_transcript(
    transcript_segments: List[Dict],
    max_tokens: int = 400,
    min_tokens: int = 100,
    overlap_tokens: int = 50
) -> List[Dict]:

    if not transcript_segments:
        return []

    # -------------------------
    # CLEAN + FILTER
    # -------------------------
    cleaned = []
    for seg in transcript_segments:
        text = _clean_text(seg.get("text", ""))

        if text and not _is_noise(text):
            cleaned.append({
                **seg,
                "text": text,
                "_idx": len(cleaned)
            })

    if not cleaned:
        return []

    texts = [seg["text"] for seg in cleaned]
    embeddings = _model.encode(texts, normalize_embeddings=True)

    domain = _detect_domain(texts)
    logger.info(f"Detected domain: {domain}")

    similarities = [
        float(np.dot(embeddings[i - 1], embeddings[i]))
        for i in range(1, len(embeddings))
    ]

    threshold = _adaptive_threshold(similarities)

    chunks = []
    current_chunk = []
    current_embeddings = []
    current_tokens = 0

    for i, seg in enumerate(cleaned):
        seg_tokens = _estimate_tokens(seg["text"])
        speaker = _get_speaker(seg)

        if current_chunk:
            chunk_emb = np.mean(current_embeddings, axis=0)
            sim = float(np.dot(chunk_emb, embeddings[i]))

            speaker_change = speaker != _get_speaker(current_chunk[-1])

            domain_split = False
            if domain == "podcast" and speaker_change:
                domain_split = True
            elif domain == "technical" and sim < threshold:
                domain_split = True

            should_split = (
                (current_tokens + seg_tokens > max_tokens)
                or (sim < threshold)
                or domain_split
            )

            if should_split:
                # 🔥 FORCE MEANINGFUL SIZE
                if current_tokens < min_tokens * 1.5:
                    current_chunk.append(seg)
                    current_embeddings.append(embeddings[i])
                    current_tokens += seg_tokens
                    continue

                chunk_text = " ".join([s["text"] for s in current_chunk])

                chunks.append({
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "start": current_chunk[0]["start"],
                    "end": current_chunk[-1]["start"] + current_chunk[-1].get("duration", 0),
                    "segments": current_chunk,
                    "token_count": current_tokens
                })

                # -------------------------
                # OVERLAP (SAFE)
                # -------------------------
                overlap = []
                tokens = 0

                for s in reversed(current_chunk):
                    t = _estimate_tokens(s["text"])
                    if tokens + t > overlap_tokens:
                        break
                    overlap.insert(0, s)
                    tokens += t

                current_chunk = overlap.copy()
                current_embeddings = [embeddings[s["_idx"]] for s in current_chunk]
                current_tokens = sum(_estimate_tokens(s["text"]) for s in current_chunk)

        current_chunk.append(seg)
        current_embeddings.append(embeddings[i])
        current_tokens += seg_tokens

    # -------------------------
    # FINAL CHUNK
    # -------------------------
    if current_chunk:
        chunk_text = " ".join([s["text"] for s in current_chunk])

        chunks.append({
            "chunk_id": len(chunks),
            "text": chunk_text,
            "start": current_chunk[0]["start"],
            "end": current_chunk[-1]["start"] + current_chunk[-1].get("duration", 0),
            "segments": current_chunk,
            "token_count": current_tokens
        })

    # -------------------------
    # 🔥 SUMMARIES + CITATIONS
    # -------------------------
    summaries = _batch_summarize(chunks)

    for chunk in chunks:
        key = _hash_text(chunk["text"])
        chunk["summary"] = summaries[key]["summary"]
        chunk["citations"] = summaries[key]["citations"]

    logger.info(f"Chunking completed: {len(chunks)} high-quality chunks")

    return chunks