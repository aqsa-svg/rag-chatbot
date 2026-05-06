# ConvoRAG — Conversation Intelligence Chatbot

A full RAG (Retrieval-Augmented Generation) system built on 191,578 conversation messages across 11,000 conversations. No ChatGPT dependency — pure Python TF-IDF retrieval.

---

## 🏗 Architecture Overview

```
conversations.csv
      │
      ▼
 data_loader.py          ← Parses CSV into flat chronological messages
      │
      ▼
 build_index.py          ← One-time index builder (run before server)
  ├── rag_engine.py      ← TF-IDF fit + topic detection + checkpointing
  └── persona_extractor  ← Pattern-based persona extraction
      │
      ▼
   data/index/           ← Saved vectorizer + checkpoints (32 MB)
   data/persona.json     ← Structured persona JSON
      │
      ▼
    app.py               ← Flask REST API
      │
  frontend/index.html    ← Chatbot UI (tabs: Chat, Topics, Checkpoints)
```

---

## 🚀 Quick Start

> Run all commands from the `rag-chatbot` directory.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the data
If your CSV is not already in `data/conversations.csv`, copy it there:
```bash
mkdir -p data
copy conversations.csv data\conversations.csv
```

### 3. Build the index (one-time, ~90 seconds)
This generates `data/messages.json`, `data/index/index.pkl`, and `data/persona.json`.
```bash
python build_index.py
```

### 4. Start the server
```bash
python app.py
```

### 5. Open the UI
```text
http://localhost:5000
```

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                    # Flask API server
├── build_index.py            # Index builder script
├── requirements.txt
├── backend/
│   ├── data_loader.py        # CSV parser
│   ├── rag_engine.py         # TF-IDF, topic detection, checkpoints
│   ├── persona_extractor.py  # Pattern-based persona extraction
│   └── answer_generator.py   # Template + RAG answer generation
├── frontend/
│   └── index.html            # Full chatbot UI
└── data/
    ├── conversations.csv     # Input data
    ├── messages.json         # Parsed flat message list
    ├── persona.json          # Extracted persona
    └── index/
        ├── index.pkl         # Vectorizer + checkpoints (with sparse vectors)
        └── checkpoints.json  # Human-readable checkpoints (no vectors)
```

---

## 🔍 How Topic Detection Works

### Strategy: Sliding Window Cosine Similarity Drop

Topic detection happens **within each conversation** (each "day" in the CSV is one independent conversation).

**Algorithm:**
1. Fit a TF-IDF vectorizer on all 191,578 messages
2. For each conversation, slide a window of `W=4` messages
3. At each position, compare the current window's TF-IDF vector to the previous window
4. If **cosine similarity < 0.10**, a topic boundary is inserted
5. Minimum segment size = 4 messages (prevents micro-segmentation)

**Why per-conversation?**  
The dataset has 11,000 independent two-person conversations. Topics shift within each conversation (e.g., greeting → hobbies → family → travel). Detecting shifts globally across conversations would just find day-to-day content variation, not meaningful topic changes within a dialogue.

**Result:** 29,607 topic segments across 11,000 conversations (avg ~2.7 segments per conversation).

**Output format:**
```
Topic: Day 2218 · Seg 4  →  messages 8–12  →  [music, bands, favorite, rock]
Topic: Day 2218 · Seg 5  →  messages 13–16 →  [family, kids, weekend, park]
```

---

## 📦 How Retrieval Works

### Dual-index retrieval: Topics + Message Chunks

When a user asks a question, the system retrieves from **two independent indexes**:

#### Index 1: Topic Checkpoints (29,607 entries)
- Each entry = one detected topic segment
- Stored as **sparse TF-IDF vector** (non-zero indices + values)
- Scored via sparse cosine similarity against the query vector

#### Index 2: 100-Message Checkpoints (1,916 entries)  
- Every 100 consecutive messages in chronological order
- Independent of topics — purely time-based
- Useful for queries that need broader temporal context

#### Retrieval pipeline:
```
Query → clean_text() → TF-IDF transform → dense query vector
                                                    │
        ┌───────────────────────────────────────────┤
        │                                           │
  Topic index                              Chunk index
  sparse cosine × 29607                  sparse cosine × 1916
        │                                           │
  Top 5 topics                            Top 3 chunks
        │
  Pull raw messages from best 2 topics
        │
  Build context string
        │
  Generate answer (template + persona)
```

---

## 👤 How Persona Is Built

The persona is extracted using **pattern matching on actual conversation signals** — nothing is inferred or guessed.

### Four dimensions:

#### 1. Habits (16 categories)
Regular expressions scan every message for habit keywords:
- `coffee_drinker`: `/coffee|espresso|latte|caffeine/`  
- `gamer`: `/game|gaming|level up|xbox|playstation/`  
- Requires ≥2 mentions to be counted as a detected habit

#### 2. Personality Traits (12 categories)
Keyword patterns scored as mentions per 100 messages:
- `humorous`: `/haha|lol|lmao|funny|joke|jk/`
- `emotional`: `/feel|hurt|cry|sad|excited|anxious/`
- `analytical`: `/because|therefore|logically|actually|precisely/`

#### 3. Communication Style
Per-message binary features (% of messages with each):
- emoji usage, caps emphasis, abbreviations (lol/tbh/imo), exclamation rate, question rate

#### 4. Personal Facts
Regex extraction of structured facts:
- Occupation: `"I'm a [X]"`, `"work as a [X]"`
- Location: `"from/in/living in [City]"`
- Relationships: `"my wife/husband/partner/kids"`
- Hobbies: `"I love/enjoy [X]"`, `"I'm into [X]"`

**Everything is grounded in text evidence, not assumptions.**

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Main chatbot endpoint |
| `/api/retrieve` | POST | Raw retrieval results |
| `/api/persona` | GET | Full persona JSON |
| `/api/stats` | GET | System stats + topic preview |
| `/api/topics` | GET | Paginated topic checkpoints |
| `/api/checkpoints` | GET | Paginated 100-msg checkpoints |
| `/api/health` | GET | Health check |

### POST /api/chat
```json
// Request
{ "query": "What are their habits?" }

// Response
{
  "query": "...",
  "answer": "## 🔄 User Habits\n...",
  "sources": {
    "topics": [{"label": "Day 2218 · Seg 4", "score": 0.469, "keywords": [...]}],
    "chunks": [{"range": "53501–53600", "score": 0.096}]
  }
}
```

---

## ☁️ Cloud Deployment (Render)

1. Push repo to GitHub
2. Create new Web Service on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt && python build_index.py`
4. Set start command: `python app.py`
5. Add env var: `PORT=10000`

**Note:** First deploy will take ~3 minutes to build the index.

---

## 🐳 Docker

```bash
docker build -t convorag .
docker run -p 5000:5000 convorag
```

---

## 📊 Stats on the Dataset

| Metric | Value |
|--------|-------|
| Total conversations | 11,000 |
| Total messages | 191,578 |
| Avg messages/conversation | 17.4 |
| Detected topic segments | 29,607 |
| Avg segments/conversation | 2.7 |
| 100-message chunks | 1,916 |
| TF-IDF vocabulary | 8,000 tokens |
| Index size | 32 MB |
| Index build time | ~90s |
| Query latency | <200ms |
