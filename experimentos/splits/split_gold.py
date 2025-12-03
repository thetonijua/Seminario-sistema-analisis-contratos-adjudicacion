
import json, random, os
from pathlib import Path

random.seed(42)

GOLD = "data/actas_json/metadata_gold.jsonl"
OUT_DIR = Path("experimentos/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

items = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
random.shuffle(items)

# 80/20
n = len(items)
cut = max(1, int(0.8 * n))
train, test = items[:cut], items[cut:]

with open(OUT_DIR/"gold_train.jsonl", "w", encoding="utf-8") as f:
    for x in train: f.write(json.dumps(x, ensure_ascii=False)+"\n")

with open(OUT_DIR/"gold_test.jsonl", "w", encoding="utf-8") as f:
    for x in test: f.write(json.dumps(x, ensure_ascii=False)+"\n")

print(f"[OK] train={len(train)} test={len(test)}")

