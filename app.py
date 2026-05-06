"""
app.py — ConvoRAG Flask API
"""
import os, sys, json, re, time, pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter
import numpy as np

# ── clean_text must be defined BEFORE unpickling ─────────────
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

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(t for t in text.split() if t not in STOPWORDS and len(t) > 2)

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "clean_text":
            return clean_text
        return super().find_class(module, name)

# ── PERSONA / ANSWER GEN ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from persona_extractor import load_persona
from answer_generator import AnswerGenerator

INDEX_DIR     = os.path.join("data", "index")
PERSONA_PATH  = os.path.join("data", "persona.json")
MESSAGES_PATH = os.path.join("data", "messages.json")

# ── STARTUP ───────────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}. "
            f"Run `python build_index.py` from the repository root first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


print("[Server] Loading index...")
t0 = time.time()

index_path = os.path.join(INDEX_DIR, "index.pkl")
if not os.path.exists(index_path):
    raise FileNotFoundError(
        f"Index not found: {index_path}. "
        f"Run `python build_index.py` from the repository root first."
    )

with open(index_path, "rb") as f:
    idx_data = SafeUnpickler(f).load()

MESSAGES = load_json(MESSAGES_PATH)

VECTORIZER  = idx_data["vectorizer"]
CHECKPOINTS = idx_data["checkpoints"]
PERSONA     = load_persona(PERSONA_PATH)
GENERATOR   = AnswerGenerator()

MSG_ID_MAP = {m["id"]: i for i, m in enumerate(MESSAGES)}

print(f"[Server] Ready in {time.time()-t0:.1f}s — "
      f"{CHECKPOINTS['total_topics']} topics, "
      f"{CHECKPOINTS['total_chunks']} chunks, "
      f"{CHECKPOINTS['total_messages']:,} messages")

# ── RETRIEVAL HELPERS ─────────────────────────────────────────

def sparse_cosine(sparse_list, dense_vec):
    dot = sum(dense_vec[i] * v for i, v in sparse_list if i < len(dense_vec))
    n1  = sum(v * v for _, v in sparse_list) ** 0.5
    n2  = np.linalg.norm(dense_vec)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def do_retrieve(query, top_k_topics=5, top_k_chunks=3):
    q_vec = VECTORIZER.transform([clean_text(query)]).toarray()[0]

    topic_scores = sorted(
        ((sparse_cosine(tc["vector"], q_vec), i)
         for i, tc in enumerate(CHECKPOINTS["topic_checkpoints"])),
        reverse=True
    )
    chunk_scores = sorted(
        ((sparse_cosine(mc["vector"], q_vec), i)
         for i, mc in enumerate(CHECKPOINTS["message_checkpoints"])),
        reverse=True
    )

    top_topics = [
        {k: v for k, v in CHECKPOINTS["topic_checkpoints"][i].items() if k != "vector"}
        | {"score": float(score)}
        for score, i in topic_scores[:top_k_topics]
    ]
    top_chunks = [
        {k: v for k, v in CHECKPOINTS["message_checkpoints"][i].items() if k != "vector"}
        | {"score": float(score)}
        for score, i in chunk_scores[:top_k_chunks]
    ]

    raw_messages = []
    seen = set()
    for tc in top_topics[:2]:
        s_pos = tc.get("start_idx")
        e_pos = tc.get("end_idx")
        if s_pos is None or e_pos is None:
            s_pos = MSG_ID_MAP.get(tc.get("start_msg_id", 0), 0)
            e_pos = MSG_ID_MAP.get(tc.get("end_msg_id", 0), s_pos + 15)
        for m in MESSAGES[s_pos: min(e_pos + 1, s_pos + 20)]:
            if m["id"] not in seen:
                raw_messages.append(m)
                seen.add(m["id"])

    context_text = "\n\n".join(
        f"[{tc['label']}]\nKeywords: {', '.join(tc['keywords'])}\n{tc['summary']}"
        for tc in top_topics
    )

    return {"top_topics": top_topics, "top_chunks": top_chunks,
            "raw_messages": raw_messages[:20], "context_text": context_text}

# ── FLASK APP ─────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok",
                    "topics": CHECKPOINTS["total_topics"],
                    "messages": CHECKPOINTS["total_messages"]})

@app.route("/api/stats")
def stats():
    preview = [
        {"topic_id": tc["topic_id"], "label": tc["label"],
         "message_count": tc["message_count"], "keywords": tc["keywords"],
         "summary_snippet": tc["summary"][:100]}
        for tc in CHECKPOINTS["topic_checkpoints"][:30]
    ]
    return jsonify({"total_messages": CHECKPOINTS["total_messages"],
                    "total_topics": CHECKPOINTS["total_topics"],
                    "total_chunks": CHECKPOINTS["total_chunks"],
                    "topics_preview": preview})

@app.route("/api/chat", methods=["POST"])
def chat():
    data  = request.get_json()
    query = (data or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    retrieved = do_retrieve(query)
    answer    = GENERATOR.answer(query, retrieved, PERSONA)
    return jsonify({
        "query": query,
        "answer": answer,
        "sources": {
            "topics": [{"label": t["label"], "score": round(t["score"], 3),
                        "keywords": t["keywords"]} for t in retrieved["top_topics"][:3]],
            "chunks": [{"range": c["message_range"], "score": round(c["score"], 3)}
                       for c in retrieved["top_chunks"]]
        }
    })

@app.route("/api/retrieve", methods=["POST"])
def retrieve_api():
    data  = request.get_json()
    query = (data or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    r = do_retrieve(query, top_k_topics=5, top_k_chunks=5)
    return jsonify({k: v for k, v in r.items() if k != "context_text"})

@app.route("/api/persona")
def persona():
    return jsonify(PERSONA)

@app.route("/api/topics")
def topics():
    tcs      = CHECKPOINTS["topic_checkpoints"]
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    start    = (page - 1) * per_page
    result   = [{k: v for k, v in tc.items() if k != "vector"}
                for tc in tcs[start: start + per_page]]
    return jsonify({"page": page, "total": len(tcs), "topics": result})

@app.route("/api/checkpoints")
def checkpoints():
    mcs      = CHECKPOINTS["message_checkpoints"]
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    start    = (page - 1) * per_page
    result   = [{k: v for k, v in mc.items() if k != "vector"}
                for mc in mcs[start: start + per_page]]
    return jsonify({"page": page, "total": len(mcs), "checkpoints": result})

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
