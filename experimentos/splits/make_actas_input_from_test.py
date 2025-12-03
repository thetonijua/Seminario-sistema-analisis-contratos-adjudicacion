
import json
from pathlib import Path

GOLD_TEST = "experimentos/splits/gold_test.jsonl"
ACTAS     = "data/actas_limpias_goldset.jsonl"
OUT       = "preprocesamiento/actas_limpias_goldTEST.jsonl"

test_fns = {json.loads(l).get("filename") for l in open(GOLD_TEST, encoding="utf-8")}
with open(OUT, "w", encoding="utf-8") as out:
    for line in open(ACTAS, encoding="utf-8"):
        if not line.strip(): continue
        j = json.loads(line)
        if j.get("archivo") in test_fns or j.get("filename") in test_fns:
            out.write(json.dumps(j, ensure_ascii=False)+"\n")
print("[OK] escrito:", OUT)
