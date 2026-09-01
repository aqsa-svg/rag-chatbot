# ConvoRAG — Conversation Intelligence Chatbot

**Ask questions about 191,578 real chat messages and get an answer built from the messages themselves — retrieval is pure scikit-learn, so the only thing the LLM does is phrase the reply.**

<!-- SCREENSHOT: the chat UI answering a question with its retrieved messages shown.
     Replace with: ![ConvoRAG UI](docs/screenshot-chat.png) -->
> 📸 *Screenshot placeholder — chat UI with an answer and the messages it was drawn from.*

> **No hosted demo.** The built index is ~331 MB, which is why it isn't deployed. Runs locally; setup verified below.

---

## The problem

Two years of chat history is a corpus you can't read and can't search usefully. Keyword search finds the message with the word in it, not the conversation that answers the question. Feeding it all to an LLM isn't an option either — 191,578 messages don't fit in a context window, and paying to embed them is a real cost.

So the constraint here was deliberate: **retrieve with classical IR, generate with an LLM, and never let the LLM near the corpus.** TF-IDF plus cosine similarity over 20,000 features costs nothing to build, runs offline, and is fully inspectable — you can always see exactly which messages produced an answer.

## What it does

- **Retrieves over 191,578 messages** using a TF-IDF index (1–2 grams, 20k features, `min_df=3`) with cosine similarity — no embeddings, no vector database, no API calls.
- **Chunks the corpus into 1,916 hundred-message checkpoints** so retrieval returns coherent stretches of conversation instead of isolated lines.
- **Extracts a structured persona** from the corpus by regex pattern banks — habits, personality traits, and personal facts — computed without any model.
- **Generates the final answer with Gemini** from the retrieved messages only.
- **Serves it over Flask** with a single-page frontend.

## Design decisions and tradeoffs

**Classical IR instead of embeddings.** TF-IDF over 191,578 messages builds in 162 seconds on a laptop, costs nothing, needs no key, and every score is traceable to specific terms. Dense embeddings would retrieve better on paraphrase — someone asking "did we ever talk about my job" won't match a message saying "work has been rough" — and that's a genuine, accepted quality ceiling. In exchange the index is reproducible offline by anyone who clones the repo.

**Extractive summarisation before generation.** `extractive_summary()` picks the messages closest to the cluster centroid rather than sending everything retrieved. It cuts prompt size and keeps the answer anchored to real messages, at the cost of dropping context that a longer prompt might have used.

**The LLM is confined to phrasing.** Retrieval, chunking, topic segmentation, and persona extraction are all deterministic Python. Only the final sentence generation calls Gemini. That makes almost the entire system testable and debuggable without a key, and means an API outage degrades the wording rather than the search.

**Persona extraction by regex pattern banks, not by asking a model.** Deterministic, free, and auditable — you can read the pattern that fired. It's also brittle in the way regex always is, and it takes 347 seconds over this corpus, which is longer than building the entire retrieval index.

## Results

Measured by running `build_index.py` end to end on this repo's committed CSV, Python 3.11.0 in a clean venv.

| Measured | Value |
|---|---|
| Corpus | **191,578 messages** across **11,000 conversations** |
| TF-IDF index | 20,000 max features, 1–2 grams, `min_df=3` |
| 100-message checkpoints | **1,916** |
| Topics detected | **1** — see the problem below |
| Index build time | **162.1 s** |
| Persona extraction time | **347.5 s** |
| Built index size | **331 MB** (`data/index/index.pkl`) |
| Persona output | 5 habits, 5 dominant traits |
| Source size | 1,105 lines Python |

Persona actually extracted from this corpus: habits `music_lover, reader, foodie, pet_owner, movie_watcher`; dominant traits `caring, extroverted, creative, emotional, analytical`.

### Topic detection currently produces one topic

This is the honest headline. `TopicDetector` is configured `window_size=15, step=8, threshold=0.12, min_topic_size=10` and, run over this corpus, it finds **1 topic** — meaning it detects no boundaries at all and the feature contributes nothing. Two causes, both real:

1. **It runs over the flattened message stream, not per conversation.** `CheckpointBuilder.build()` calls `self.detector.detect(messages, ...)` once on the concatenation of all 11,000 conversations. Adjacent 15-message windows drawn from a 191k-message stream are almost always similar enough to clear a 0.12 threshold, so no boundary ever fires.
2. **The threshold is likely too permissive** even for per-conversation use.

Retrieval still works — it rests on the 1,916 message checkpoints, not on topics — so this is a dead feature rather than a broken system.

- **[TODO]** Move detection inside the per-conversation loop (group by the `day` field, which `data_loader` already provides) and re-measure the topic count.
- **[TODO]** Sweep `threshold` from 0.12 to 0.6 and report topics-per-conversation, then pick a value from the curve.
- **[TODO]** Retrieval quality is unmeasured. There is no test set. *To measure:* write 20–30 questions with the message ids that should be retrieved, and report recall@5 and MRR. Without it, "retrieval works" is an assertion.

### Two bugs found by running it

- **`build_index.py` exits non-zero after succeeding.** Every artifact is written correctly, then the final `print(f"\n✅ All done!...")` raises `UnicodeEncodeError: 'charmap' codec can't encode character '✅'` on a Windows `cp1252` console. The build is fine; the script's exit code is not, which would fail any CI step. Fix: drop the emoji or set `PYTHONIOENCODING=utf-8`.
- **The index is 331 MB**, which rules out most free hosting and is why there's no live demo.

## Architecture

```
  data/conversations.csv
  11,000 conversations  |  191,578 messages
             |
             v
  backend/data_loader.py   -> flat message list
                              {id, day, sender, text, msg_in_day}
             |
             +----------------------------+
             |                            |
             v                            v
  TfidfVectorizer                backend/persona_extractor.py
  max_features=20000             regex pattern banks -> habits,
  ngram_range=(1,2)              traits, personal facts
  min_df=3                       (347.5 s, no model calls)
             |                            |
             v                            v
  TopicDetector                    data/persona.json
  window=15 step=8 thr=0.12
  -> 1 topic  (see above)
             |
             v
  100-message checkpoints  -> 1,916 chunks
             |
             v
  data/index/index.pkl  (331 MB)  +  checkpoints.json
             |
             |  QUERY
             v
  cosine_similarity(query, chunks) -> top matches
             |
             v
  extractive_summary()  centroid-nearest messages
             |
             v
  backend/answer_generator.py  ->  Gemini  ->  answer
             |
             v
  app.py (Flask)  ->  frontend/index.html
```

## Stack

Python · scikit-learn (TF-IDF, cosine similarity) · NumPy · Flask + flask-cors · google-generativeai (answer phrasing only) · Docker

## Running locally

Verified from a clean venv on Python 3.11.0:

```bash
git clone https://github.com/aqsa-svg/rag-chatbot.git
cd rag-chatbot

py -3.11 -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Build the index — no API key needed. ~8.5 minutes total,
# and it writes ~360 MB into data/.
export PYTHONIOENCODING=utf-8   # avoids the cp1252 crash on the final print
python build_index.py

echo "GEMINI_API_KEY=your_key_here" > .env   # only needed for answer generation
python app.py                                # http://localhost:5000
```

Timings measured on this machine: index 162.1 s, persona 347.5 s. Everything except the final answer generation runs with no key.

## Known limitations

- **Topic detection finds 1 topic** and currently contributes nothing. See above.
- **Retrieval quality is unmeasured** — no test set, no recall@k, no MRR.
- **TF-IDF cannot match paraphrase.** A question worded differently from the messages it should find will miss. This is the accepted cost of the no-embeddings decision, not an oversight.
- **`build_index.py` exits non-zero** on Windows despite completing successfully.
- **331 MB index, ~360 MB of build artifacts.** Correctly gitignored, but it means no free-tier deployment and a slow cold start.
- **Persona extraction is regex-based**, so it detects only the patterns someone thought to write, and takes twice as long as building the retrieval index.
- **Single fixed corpus.** No upload path; a new dataset means replacing `data/conversations.csv` and rebuilding.
- **No tests.** Nothing here is covered by an automated check.
- **The corpus is a public conversation dataset**, not private data — worth stating, since a persona extractor pointed at real chat logs is a privacy-sensitive tool.

## Project layout

```
build_index.py               one-shot builder: load -> TF-IDF -> checkpoints -> persona
backend/data_loader.py       CSV -> flat message list
backend/rag_engine.py        TF-IDF fit, TopicDetector, CheckpointBuilder, retrieval
backend/persona_extractor.py regex pattern banks -> persona JSON
backend/answer_generator.py  Gemini call, retrieved messages only
app.py                       Flask API
frontend/index.html          single-page chat UI
data/conversations.csv       the corpus (committed)
data/index/, data/*.json     build artifacts (gitignored)
Dockerfile
```

---

*The retrieval half is deliberately boring — TF-IDF, cosine similarity, no embeddings — and it's the half that works. The topic detector is the interesting failure, and it's documented above rather than quietly left in the feature list.*
