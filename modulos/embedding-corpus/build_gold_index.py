# modulos/embedding-corpus/build_gold_index.py
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path

# === Paths ===
# Carpeta donde está este script (modulos/embedding-corpus)
HERE = Path(__file__).resolve().parent

# Entrada: JSONL con el gold set
METADATA_PATH = Path("data/actas_json/metadata_gold.jsonl")  # ajusta si fuera necesario

# Salidas: en la misma carpeta del script
INDEX_PATH = HERE / "gold.index"
MAPPING_PATH = HERE / "gold_mapping.jsonl"

# === Cargar modelo de embeddings  ===
model = SentenceTransformer("paraphrase-mpnet-base-v2")

# === Cargar resoluciones con riesgos (gold set) ===
docs = []
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            docs.append(json.loads(line))

if not docs:
    raise RuntimeError(f"No se encontraron documentos en {METADATA_PATH}")

# === Preparar textos base para el embedding ===
# Usamos una representación simple: nota_curador + etiquetas de riesgo
corpus_texts = []
for d in docs:
    nota = d.get("nota_curador", "")
    riesgos = " ".join(d.get("riesgos", []))
    text = (nota + " " + riesgos).strip()
    if not text:
        # fallback mínimo para no dejar vacío
        text = d.get("doc_id", "sin_doc_id")
    corpus_texts.append(text)

# === Generar embeddings ===
embeddings = model.encode(corpus_texts, convert_to_numpy=True)  # float32

# === Indexar en FAISS (L2) ===
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === Guardar índice en la misma carpeta del script ===
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX_PATH))

# === Guardar mapping (idx -> metadatos del doc) en la misma carpeta ===
with open(MAPPING_PATH, "w", encoding="utf-8") as f:
    for i, d in enumerate(docs):
        # Incluimos el índice interno y algunos campos clave
        row = {
            "idx": i,
            "doc_id": d.get("doc_id"),
            "filepath": d.get("filepath"),
            "filename": d.get("filename"),
            "riesgos": d.get("riesgos", []),
            "nota_curador": d.get("nota_curador", "")
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[OK] Índice guardado en: {INDEX_PATH}")
print(f"[OK] Mapping guardado en: {MAPPING_PATH}")
print(f"[INFO] Documentos indexados: {len(docs)}")
