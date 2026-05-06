"""
rag_engine.py
─────────────
Pure-Python RAG engine. No external LLM needed for embeddings.

Responsibilities:
  1. TF-IDF vectorization (sklearn for speed on 191k messages)
  2. Topic detection: sliding-window cosine similarity drop
  3. Topic checkpoints (per topic segment)
  4. 100-message checkpoints (independent of topics)
  5. Retrieval: query → top topic summaries + raw message chunks
"""

import re
import json
import pickle
import os
from typing import List, Dict, Optional
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","is","it","in","on","at","to","for","of","and","or",
    "but","not","with","this","that","was","are","be","have","do","i",
    "you","he","she","we","they","my","your","his","her","our","their",
    "so","just","like","what","how","when","where","who","which","then",
    "them","its","been","has","had","will","would","could","should","can",
    "did","does","am","from","up","out","if","about","as","me","him","us",
    "oh","ok","okay","yeah","yes","no","hi","hey","hello","well","good",
    "great","nice","sure","right","think","know","get","got","really",
    "too","also","want","going","go","see","say","said","make","way","one",
    "thing","things","time","day","want","need","feel","felt","lot","much",
    "now","never","always","still","very","even","just","ll","ve","re","m","s","t","d"
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def extractive_summary(texts: List[str], vectorizer, n: int = 4) -> str:
    """Pick top-N sentences by TF-IDF centroid similarity."""
    sentences = []
    for t in texts:
        sentences.extend(re.split(r"(?<=[.!?])\s+", t.strip()))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    if not sentences:
        return " ".join(texts)[:400]
    if len(sentences) <= n:
        return " ".join(sentences)

    try:
        clean_sents = [clean_text(s) for s in sentences]
        vecs = vectorizer.transform(clean_sents)
        centroid = vecs.mean(axis=0)
        scores = cosine_similarity(vecs, centroid).flatten()
        top_idx = np.argsort(scores)[::-1][:n]
        top_idx_sorted = sorted(top_idx)
        return " ".join(sentences[i] for i in top_idx_sorted)
    except Exception:
        return " ".join(sentences[:n])


# ──────────────────────────────────────────────────────────────
# TOPIC DETECTOR
# ──────────────────────────────────────────────────────────────

class TopicDetector:
    """
    Detects topic boundaries by comparing overlapping windows of messages.
    A boundary is inserted when cosine similarity between adjacent windows
    drops below `threshold`.
    """

    def __init__(self, window_size: int = 15, step: int = 8,
                 threshold: float = 0.12, min_topic_size: int = 10):
        self.window_size = window_size
        self.step = step
        self.threshold = threshold
        self.min_topic_size = min_topic_size

    def detect(self, messages: List[Dict], vectorizer) -> List[int]:
        """
        Returns sorted list of boundary indices where new topics begin.
        Always starts with 0.
        """
        texts = [clean_text(m["text"]) for m in messages]
        N = len(texts)

        if N < self.window_size * 2:
            return [0]

        vecs = vectorizer.transform(texts)
        boundaries = [0]
        prev_window_vec = None

        for i in range(0, N - self.window_size, self.step):
            window_vec = np.asarray(vecs[i: i + self.window_size].mean(axis=0))

            if prev_window_vec is not None:
                sim = float(cosine_similarity(window_vec, prev_window_vec)[0, 0])
                if sim < self.threshold:
                    candidate = i
                    if candidate - boundaries[-1] >= self.min_topic_size:
                        boundaries.append(candidate)

            prev_window_vec = window_vec

        return boundaries


# ──────────────────────────────────────────────────────────────
# CHECKPOINT BUILDER
# ──────────────────────────────────────────────────────────────

class CheckpointBuilder:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=3,
            preprocessor=clean_text
        )
        self.detector = TopicDetector(
            window_size=15, step=8, threshold=0.12, min_topic_size=10
        )
        self.checkpoints: Optional[Dict] = None

    def build(self, messages: List[Dict]) -> Dict:
        print(f"[RAG] Fitting TF-IDF on {len(messages)} messages...")
        texts = [m["text"] for m in messages]
        self.vectorizer.fit(texts)

        # ── Topic Checkpoints ──────────────────────────────────
        print("[RAG] Detecting topic boundaries...")
        boundaries = self.detector.detect(messages, self.vectorizer)
        boundaries.append(len(messages))  # sentinel

        print(f"[RAG] Found {len(boundaries)-1} topics")

        topic_checkpoints = []
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            seg_msgs = messages[start:end]
            seg_texts = [m["text"] for m in seg_msgs]
            seg_clean = [clean_text(t) for t in seg_texts]

            summary = extractive_summary(seg_texts, self.vectorizer, n=4)
            keywords = self._top_keywords(seg_clean, n=8)
            vec = self.vectorizer.transform([" ".join(seg_clean)]).toarray()[0]

            topic_checkpoints.append({
                "topic_id": idx + 1,
                "label": f"Topic {idx + 1}",
                "start_idx": start,
                "end_idx": end - 1,
                "message_range": f"{start + 1}–{end}",
                "start_day": messages[start]["day"],
                "end_day": messages[end - 1]["day"],
                "message_count": end - start,
                "summary": summary,
                "keywords": keywords,
                "vector": vec.tolist()
            })

        # ── 100-Message Checkpoints ────────────────────────────
        print("[RAG] Building 100-message checkpoints...")
        message_checkpoints = []
        for chunk_start in range(0, len(messages), 100):
            chunk_end = min(chunk_start + 100, len(messages))
            chunk_msgs = messages[chunk_start:chunk_end]
            chunk_texts = [m["text"] for m in chunk_msgs]
            chunk_clean = [clean_text(t) for t in chunk_texts]

            summary = extractive_summary(chunk_texts, self.vectorizer, n=5)
            keywords = self._top_keywords(chunk_clean, n=6)
            vec = self.vectorizer.transform([" ".join(chunk_clean)]).toarray()[0]

            message_checkpoints.append({
                "chunk_id": len(message_checkpoints) + 1,
                "start_idx": chunk_start,
                "end_idx": chunk_end - 1,
                "message_range": f"{chunk_start + 1}–{chunk_end}",
                "start_day": messages[chunk_start]["day"],
                "end_day": messages[chunk_end - 1]["day"],
                "summary": summary,
                "keywords": keywords,
                "vector": vec.tolist()
            })

        self.checkpoints = {
            "topic_checkpoints": topic_checkpoints,
            "message_checkpoints": message_checkpoints,
            "total_messages": len(messages),
            "total_topics": len(topic_checkpoints),
            "total_chunks": len(message_checkpoints)
        }
        print(f"[RAG] Done. {len(topic_checkpoints)} topic CPs, "
              f"{len(message_checkpoints)} 100-msg CPs")
        return self.checkpoints

    def _top_keywords(self, clean_texts: List[str], n: int) -> List[str]:
        combined = " ".join(clean_texts)
        freq = Counter(combined.split())
        return [w for w, _ in freq.most_common(n)]


# ──────────────────────────────────────────────────────────────
# RETRIEVER
# ──────────────────────────────────────────────────────────────

class Retriever:

    def __init__(self, checkpoints: Dict, messages: List[Dict],
                 vectorizer):
        self.checkpoints = checkpoints
        self.messages = messages
        self.vectorizer = vectorizer

        self._topic_vecs = np.array(
            [tc["vector"] for tc in checkpoints["topic_checkpoints"]]
        )
        self._chunk_vecs = np.array(
            [mc["vector"] for mc in checkpoints["message_checkpoints"]]
        )

    def retrieve(self, query: str, top_k_topics: int = 3,
                 top_k_chunks: int = 3) -> Dict:
        q_vec = self.vectorizer.transform([query]).toarray()

        topic_sims = cosine_similarity(q_vec, self._topic_vecs)[0]
        top_topic_idx = np.argsort(topic_sims)[::-1][:top_k_topics]
        top_topics = [
            {**{k: v for k, v in self.checkpoints["topic_checkpoints"][i].items() if k != "vector"},
             "score": float(topic_sims[i])}
            for i in top_topic_idx
        ]

        chunk_sims = cosine_similarity(q_vec, self._chunk_vecs)[0]
        top_chunk_idx = np.argsort(chunk_sims)[::-1][:top_k_chunks]
        top_chunks = [
            {**{k: v for k, v in self.checkpoints["message_checkpoints"][i].items() if k != "vector"},
             "score": float(chunk_sims[i])}
            for i in top_chunk_idx
        ]

        raw_messages = []
        seen = set()
        for tc in top_topics[:2]:
            s, e = tc["start_idx"], tc["end_idx"]
            for m in self.messages[s: min(e + 1, s + 20)]:
                if m["id"] not in seen:
                    raw_messages.append(m)
                    seen.add(m["id"])

        context_text = "\n\n".join(
            f"[{tc['label']} | msgs {tc['message_range']}]\n{tc['summary']}"
            for tc in top_topics
        )

        return {
            "top_topics": top_topics,
            "top_chunks": top_chunks,
            "raw_messages": raw_messages[:25],
            "context_text": context_text
        }


# ──────────────────────────────────────────────────────────────
# PERSISTENCE
# ──────────────────────────────────────────────────────────────

def save_index(builder: CheckpointBuilder, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cp_json = {
        "topic_checkpoints": [
            {k: v for k, v in tc.items() if k != "vector"}
            for tc in builder.checkpoints["topic_checkpoints"]
        ],
        "message_checkpoints": [
            {k: v for k, v in mc.items() if k != "vector"}
            for mc in builder.checkpoints["message_checkpoints"]
        ],
        "total_messages": builder.checkpoints["total_messages"],
        "total_topics": builder.checkpoints["total_topics"],
        "total_chunks": builder.checkpoints["total_chunks"],
    }
    with open(os.path.join(out_dir, "checkpoints.json"), "w") as f:
        json.dump(cp_json, f, indent=2)

    with open(os.path.join(out_dir, "index.pkl"), "wb") as f:
        pickle.dump({
            "vectorizer": builder.vectorizer,
            "checkpoints": builder.checkpoints,
        }, f)

    print(f"[RAG] Index saved to {out_dir}")


def load_index(index_dir: str, messages: List[Dict]) -> Retriever:
    with open(os.path.join(index_dir, "index.pkl"), "rb") as f:
        data = pickle.load(f)
    return Retriever(data["checkpoints"], messages, data["vectorizer"])
