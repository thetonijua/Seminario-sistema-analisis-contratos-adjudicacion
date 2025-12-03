# Agrega una resolución con errores al gold set (JSONL) y copia su PDF a corpus_gold.
import argparse, json, shutil, hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path(".")
METADATA = ROOT / "data/actas_json/metadata_gold.jsonl"
CORPUS_DIR = ROOT / "data/corpus_gold/"

def sha1sum(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Ruta al PDF origen")
    ap.add_argument("--doc_id", required=True, help="ID estable para el documento")
    ap.add_argument("--riesgos", required=True, nargs="+", help="Etiquetas de riesgo (ej: MS.Considerando MR.Ley19886)")
    ap.add_argument("--evidencia", action="append", default=[], help="Cita textual (puedes repetir --evidencia)")
    ap.add_argument("--loc", action="append", default=[], help="Ubicación (pX/parrY) en mismo orden que evidencia")
    ap.add_argument("--nota", default="", help="Nota del curador")
    ap.add_argument("--validated", action="store_true", help="Marca validado=True")
    args = ap.parse_args()

    pdf_src = Path(args.pdf).resolve()
    if not pdf_src.exists():
        raise SystemExit(f"No existe el PDF: {pdf_src}")

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_dst = CORPUS_DIR / f"{args.doc_id}.pdf"

    # Evita duplicados por hash
    src_hash = sha1sum(pdf_src)
    if pdf_dst.exists():
        dst_hash = sha1sum(pdf_dst)
        if dst_hash == src_hash:
            print("[INFO] Ya existía el mismo archivo en corpus_gold (mismo hash).")
        else:
            pdf_dst = CORPUS_DIR / f"{args.doc_id}_{src_hash[:8]}.pdf"
            print(f"[WARN] Mismo doc_id con distinto hash, guardando como: {pdf_dst.name}")

    shutil.copy2(pdf_src, pdf_dst)

    # Arma evidencias
    evidencias = []
    for i, texto in enumerate(args.evidencia):
        evidencias.append({
            "texto": texto,
            "seccion": "Desconocida",
            "loc": args.loc[i] if i < len(args.loc) else "s/n"
        })

    entry = {
        "doc_id": args.doc_id,
        "filename": pdf_dst.name,
        "filepath": str(pdf_dst),
        "riesgos": args.riesgos,
        "evidencias": evidencias,
        "nota_curador": args.nota,
        "source": "LLM+RAG",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "validated": bool(args.validated)
    }

    METADATA.parent.mkdir(parents=True, exist_ok=True)
    with METADATA.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[OK] Agregado al JSONL: {METADATA}")
    print(f"[OK] Copiado PDF a: {pdf_dst}")

if __name__ == "__main__":
    main()
