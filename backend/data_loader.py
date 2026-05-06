"""
data_loader.py
Parses conversations.csv into a flat chronological list of messages.
Each row in the CSV = one conversation (one "day").
Each conversation contains lines: "User 1: ..." and "User 2: ..."
"""

import csv
import json
import os
from typing import List, Dict


def load_conversations(csv_path: str, max_days: int = None) -> List[Dict]:
    """
    Returns a flat list of messages, chronologically ordered by day.
    Each message dict:
        id        : unique int
        day       : row index (0-based)
        sender    : 'User1' or 'User2'
        text      : message text
        msg_in_day: position within the day's conversation
    """
    messages = []
    msg_id = 0

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row

        for day_idx, row in enumerate(reader):
            if max_days and day_idx >= max_days:
                break
            if not row:
                continue

            raw = row[0]
            lines = raw.split("\n")
            msg_in_day = 0

            for line in lines:
                line = line.strip()
                if line.startswith("User 1:"):
                    text = line[7:].strip()
                    sender = "User1"
                elif line.startswith("User 2:"):
                    text = line[7:].strip()
                    sender = "User2"
                else:
                    continue

                if not text:
                    continue

                messages.append({
                    "id": msg_id,
                    "day": day_idx,
                    "sender": sender,
                    "text": text,
                    "msg_in_day": msg_in_day
                })
                msg_id += 1
                msg_in_day += 1

    return messages


def save_messages(messages: List[Dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(messages)} messages to {out_path}")


def load_messages(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "..", "data", "conversations.csv")
    out_path = os.path.join(here, "..", "data", "messages.json")
    msgs = load_conversations(csv_path)
    print(f"Loaded {len(msgs)} messages from {len(set(m['day'] for m in msgs))} conversations")
    save_messages(msgs, out_path)
