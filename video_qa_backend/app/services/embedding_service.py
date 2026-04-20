import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
import logging
import pickle
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        # 🔥 BETTER MULTILINGUAL MODEL
        self.model = SentenceTransformer("intfloat/multilingual-e5-base", device=settings.get_device)

        # 🔥 STRONGER RERANKER
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=settings.get_device)

        self.dimension = 768  # e5-base dimension

    # -------------------------
    # EMBEDDINGS (E5 FORMAT)
    # -------------------------
    def generate_embedding(self, text: str) -> np.ndarray:
        text = f"query: {text}"
        return self.model.encode(text, normalize_embeddings=True)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True)

    def build_enriched_text(self, chunk: dict, video_title: str) -> str:
        if chunk.get("type") == "visual":
            return f"""
Title: {video_title}
Visual Scene Description:
{chunk.get('text', '').replace('[VISUAL] ', '')}
""".strip()
        else:
            return f"""
Title: {video_title}
Audio Transcript Summary: {chunk.get('summary', '')}
Spoken Content: {chunk.get('text', '')}
""".strip()

    # -------------------------
    # SAVE INDEX
    # -------------------------
    def save_index(self, youtube_id: str, chunks: List[dict], video_title: str):
        index_dir = Path(settings.EMBEDDINGS_DIR) / youtube_id
        index_dir.mkdir(parents=True, exist_ok=True)

        enriched_texts = [self.build_enriched_text(c, video_title) for c in chunks]
        embeddings = self.generate_embeddings(enriched_texts)

        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(index_dir / "index.faiss"))

        tokenized = [text.lower().split() for text in enriched_texts]
        bm25 = BM25Okapi(tokenized)

        metadata = {
            "chunks": chunks,
            "texts": enriched_texts,
            "embeddings": embeddings,
            "bm25": bm25
        }

        with open(index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Index saved for {youtube_id} with {len(chunks)} chunks")

    # -------------------------
    # QUERY EXPANSION (SMART)
    # -------------------------
    def expand_query(self, query: str, search_mode: str) -> str:
        if search_mode == "visual":
            return f"Visual Scene Description: {query}"
        elif search_mode == "audio":
            return f"Audio Transcript Summary: {query}"
        return query

    # -------------------------
    # HYBRID SEARCH
    # -------------------------
    def hybrid_search(
        self,
        index,
        bm25,
        texts,
        query_embedding,
        query,
        top_k=50
    ) -> List[int]:

        # Dense
        d_scores, d_indices = index.search(query_embedding, top_k)

        # BM25
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize BM25
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) + 1e-8)

        combined_scores = {}

        for i, idx in enumerate(d_indices[0]):
            combined_scores[idx] = combined_scores.get(idx, 0) + float(d_scores[0][i])

        for idx, score in enumerate(bm25_scores):
            combined_scores[idx] = combined_scores.get(idx, 0) + float(score)

        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in sorted_items[:top_k]]

    # -------------------------
    # RERANK
    # -------------------------
    def rerank(
        self,
        query: str,
        candidate_pairs: List[Tuple[int, str]],
        top_k=10
    ) -> List[int]:

        pairs = [[query, text[:300]] for _, text in candidate_pairs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidate_pairs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [idx for ((idx, _), _) in ranked[:top_k]]

    # -------------------------
    # MMR (BALANCED)
    # -------------------------
    def mmr(
        self,
        candidate_indices: List[int],
        embeddings: np.ndarray,
        query_embedding: np.ndarray,
        top_k=5,
        lambda_param=0.85
    ) -> List[int]:

        selected = []
        candidates = candidate_indices.copy()
        query_embedding = query_embedding.reshape(-1)

        while len(selected) < top_k and candidates:
            best_score = -np.inf
            best_idx = None

            for idx in candidates:
                relevance = np.dot(embeddings[idx], query_embedding)

                diversity = 0
                if selected:
                    diversity = max(
                        np.dot(embeddings[idx], embeddings[j])
                        for j in selected
                    )

                score = lambda_param * relevance - (1 - lambda_param) * diversity

                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    # -------------------------
    # FINAL SEARCH
    # -------------------------
    def search(self, youtube_id: str, query: str, top_k: int = 5, search_mode: str = "both"):
        index_dir = Path(settings.EMBEDDINGS_DIR) / youtube_id
        index_path = index_dir / "index.faiss"
        metadata_path = index_dir / "metadata.pkl"

        if not index_path.exists() or not metadata_path.exists():
            return []

        index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        chunks = metadata["chunks"]
        texts = metadata["texts"]
        embeddings = metadata.get("embeddings")

        if embeddings is None:
            logger.warning("Rebuilding embeddings (slow fallback)")
            embeddings = self.generate_embeddings(texts)

        bm25 = metadata["bm25"]

        # Query
        expanded_query = self.expand_query(query, search_mode)
        query_embedding = self.generate_embedding(expanded_query).reshape(1, -1).astype(np.float32)

        # Hybrid
        candidate_indices = self.hybrid_search(
            index,
            bm25,
            texts,
            query_embedding,
            expanded_query,
            top_k=100  # Pull more candidates initially to allow for filtering
        )

        # Filter by search_mode
        if search_mode in ["audio", "visual"]:
            filtered_indices = []
            for idx in candidate_indices:
                if chunks[idx].get("type") == search_mode:
                    filtered_indices.append(idx)
            candidate_indices = filtered_indices

        if not candidate_indices:
            return []

        # Rerank
        candidate_pairs = [(idx, texts[idx]) for idx in candidate_indices]
        reranked_indices = self.rerank(expanded_query, candidate_pairs, top_k=15)

        # MMR
        final_indices = self.mmr(
            candidate_indices=reranked_indices,
            embeddings=embeddings,
            query_embedding=query_embedding.flatten(),
            top_k=top_k
        )

        # Build results (NO HARD FILTER)
        results = []
        for idx in final_indices:
            chunk = chunks[idx]
            score = float(np.dot(embeddings[idx], query_embedding.flatten()))

            results.append({
                "chunk_id": chunk.get("chunk_id", f"vis_{idx}"),
                "text": chunk["text"],
                "score": score,
                "start": chunk.get("start"),
                "end": chunk.get("end")
            })

        # 🔥 ensure at least 1 result
        if not results and candidate_indices:
            idx = candidate_indices[0]
            chunk = chunks[idx]

            results.append({
                "chunk_id": chunk.get("chunk_id", f"vis_{idx}"),
                "text": chunk["text"],
                "score": 0.0,
                "start": chunk.get("start"),
                "end": chunk.get("end")
            })

        return results


embedding_service = EmbeddingService()