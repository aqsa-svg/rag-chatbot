"""
build_index.py
───────────────
One-time script: processes conversations.csv → builds RAG index + persona.
Run this before starting the server.

Usage:
    python build_index.py [--csv path/to/conversations.csv] [--max-days N]
"""

import sys
import os
import time
import json
import argparse

# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_conversations
from rag_engine import CheckpointBuilder, save_index
from persona_extractor import PersonaExtractor, save_persona


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join("data", "conversations.csv"),
                        help="Path to conversations CSV")
    parser.add_argument("--max-days", type=int, default=None,
                        help="Limit number of days (for testing)")
    parser.add_argument("--index-dir", default=os.path.join("data", "index"),
                        help="Where to save the RAG index")
    parser.add_argument("--persona-out", default=os.path.join("data", "persona.json"),
                        help="Where to save persona JSON")
    args = parser.parse_args()

    # ── 1. Load messages ──────────────────────────────────────
    print(f"\n[1/4] Loading conversations from {args.csv}...")
    t0 = time.time()
    messages = load_conversations(args.csv, max_days=args.max_days)
    days = len(set(m["day"] for m in messages))
    print(f"      Loaded {len(messages):,} messages from {days:,} conversations "
          f"({time.time()-t0:.1f}s)")

    # Save messages list for server use
    os.makedirs("data", exist_ok=True)
    msg_path = os.path.join("data", "messages.json")
    print(f"      Saving messages to {msg_path}...")
    with open(msg_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)
    print(f"      Saved.")

    # ── 2. Build RAG index ────────────────────────────────────
    print(f"\n[2/4] Building RAG index...")
    t1 = time.time()
    builder = CheckpointBuilder()
    checkpoints = builder.build(messages)
    print(f"      Index built in {time.time()-t1:.1f}s")
    print(f"      Topics: {checkpoints['total_topics']}")
    print(f"      100-msg chunks: {checkpoints['total_chunks']}")

    # ── 3. Save index ─────────────────────────────────────────
    print(f"\n[3/4] Saving index to {args.index_dir}...")
    save_index(builder, args.index_dir)

    # ── 4. Extract persona ────────────────────────────────────
    print(f"\n[4/4] Extracting user persona...")
    t3 = time.time()
    extractor = PersonaExtractor()
    persona = extractor.extract(messages)
    save_persona(persona, args.persona_out)
    print(f"      Persona extracted in {time.time()-t3:.1f}s")
    print(f"      Detected habits: {persona['habits']['detected'][:5]}")
    print(f"      Dominant traits: {persona['personality_traits']['dominant_traits'][:5]}")

    print(f"\n✅ All done! Total time: {time.time()-t0:.1f}s")
    print(f"   Index:   {args.index_dir}/")
    print(f"   Persona: {args.persona_out}")
    print(f"\nNow run:  python app.py")


if __name__ == "__main__":
    main()
