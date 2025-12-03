# modulos/embedding-corpus/build_gold_index.py
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path
import argparse

# === Defaults ===
HERE = Path(__file__).resolve().parent
DEFAULT_METADATA = Path("data/actas_json/metadata_gold.jsonl")
DEFAULT_INDEX = HERE / "gold.index"
DEFAULT_MAPPING = HERE / "gold_mapping.jsonl"


def build_index(metadata_path: Path, index_path: Path, mapping_path: Path):
    """Construye un índice FAISS a partir del metadata gold."""

    print(f"[INFO] Usando metadata: {metadata_path}")
    print(f"[INFO] Salidas -> index: {index_path}, mapping: {mapping_path}")

    # === Cargar modelo de embeddings ===
    model = SentenceTransformer("paraphrase-mpnet-base-v2")

    # === Cargar resoluciones con riesgos (gold set) ===
    docs = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    if not docs:
        raise RuntimeError(f"No se encontraron documentos en {metadata_path}")

    # === Preparar textos base para el embedding ===
    corpus_texts = []
    for d in docs:
        nota = d.get("nota_curador", "")
        riesgos = " ".join(d.get("riesgos", []))
        text = (nota + " " + riesgos).strip()
        if not text:
            text = d.get("doc_id", "sin_doc_id")
        corpus_texts.append(text)

    # === Generar embeddings ===
    embeddings = model.encode(corpus_texts, convert_to_numpy=True)

    # === Indexar en FAISS (L2) ===
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # === Guardar índice ===
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # === Guardar mapping ===
    with open(mapping_path, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            row = {
                "idx": i,
                "doc_id": d.get("doc_id"),
                "filepath": d.get("filepath"),
                "filename": d.get("filename"),
                "riesgos": d.get("riesgos", []),
                "nota_curador": d.get("nota_curador", "")
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Índice guardado en: {index_path}")
    print(f"[OK] Mapping guardado en: {mapping_path}")
    print(f"[INFO] Documentos indexados: {len(docs)}")


def main():
    parser = argparse.ArgumentParser(description="Construye el índice FAISS del gold set.")
    parser.add_argument("--jsonl", type=str, default=str(DEFAULT_METADATA),
                        help="Ruta al metadata_gold.jsonl (por defecto data/actas_json/metadata_gold.jsonl)")
    parser.add_argument("--out_index", type=str, default=str(DEFAULT_INDEX),
                        help="Ruta de salida del índice FAISS (por defecto gold.index)")
    parser.add_argument("--out_mapping", type=str, default=str(DEFAULT_MAPPING),
                        help="Ruta de salida del mapping (por defecto gold_mapping.jsonl)")
    args = parser.parse_args()

    build_index(Path(args.jsonl), Path(args.out_index), Path(args.out_mapping))


if __name__ == "__main__":
    main()
