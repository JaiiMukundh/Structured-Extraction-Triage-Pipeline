import json
import os
import random
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
QWEN_INPUT = BASE / "data" / "qwen" / "qwen_base.jsonl"
DISTILBERT_INPUT = BASE / "data" / "distilbert" / "distilbert_base.jsonl"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


if not QWEN_INPUT.exists() or not DISTILBERT_INPUT.exists():
    raise SystemExit("Required input files are missing.")

qwen_positives = {}
with open(QWEN_INPUT, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            record = json.loads(line)
            text = record["text"].strip()
            text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
            qwen_positives[" ".join(text.split())] = record["event_data"]

distilbert_rows = []
with open(DISTILBERT_INPUT, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            distilbert_rows.append(json.loads(line))

aligned = []
for row in distilbert_rows:
    key_text = row["text"].strip()
    key_text = key_text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    key = " ".join(key_text.split())
    if row["label"] == 1:
        aligned.append(
            {
                "text": row["text"],
                "label": 1,
                "event_data": qwen_positives.get(key),
            }
        )
    else:
        aligned.append(
            {
                "text": row["text"],
                "label": 0,
                "event_data": None,
            }
        )

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
random.shuffle(aligned)

total = len(aligned)
train_n = int(total * TRAIN_RATIO)
val_n = int(total * VAL_RATIO)

train_split = aligned[:train_n]
val_split = aligned[train_n : train_n + val_n]
test_split = aligned[train_n + val_n :]

qwen_dir = BASE / "data" / "qwen"
distilbert_dir = BASE / "data" / "distilbert"

for name, split in [("train", train_split), ("val", val_split), ("test", test_split)]:
    with open(qwen_dir / f"qwen_{name}.jsonl", "w", encoding="utf-8") as qf, open(
        distilbert_dir / f"distilbert_{name}.jsonl", "w", encoding="utf-8"
    ) as df:
        for item in split:
            qf.write(json.dumps({"text": item["text"], "event_data": item["event_data"]}, ensure_ascii=False) + "\n")
            df.write(json.dumps({"text": item["text"], "label": item["label"]}, ensure_ascii=False) + "\n")
