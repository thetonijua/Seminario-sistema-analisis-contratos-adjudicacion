# modulos/analyze_from_jsonl.py
import argparse, json, os, re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai

# ===== config =====
MODEL_EMB = "paraphrase-mpnet-base-v2"
MODEL_LLM = "gpt-3.5-turbo"
# ==================

def load_mapping(path):
    mapping = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                mapping.append(json.loads(line))
    return mapping



def pick_text(rec: dict) -> str:
   
    txt = rec.get("texto_limpio", "")
    return txt if isinstance(txt, str) else ""



def normalize_base_name(rec: dict) -> str:

    base = rec.get("archivo") or "acta"
    base = str(base).replace("\\", "/").split("/")[-1]

    # quitar extensión si existe
    if "." in base:
        base = ".".join(base.split(".")[:-1]) or base

    # limpiar caracteres raros
    base = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", base).strip("_")
    return base or "acta"



def build_prompt(tpl_path: Path, res_text: str, precedentes: list):
    tpl = Path(tpl_path).read_text(encoding="utf-8")
    lines = []
    for p in precedentes:
        lines.append(
            f"- (doc_id={p.get('doc_id')}) riesgos={p.get('riesgos')} "
            f"nota={p.get('nota_curador','')} file={p.get('filepath')}"
        )
    precedentes_str = "\n".join(lines) if lines else "- (sin precedentes encontrados)"
    prompt = tpl.replace("<<RESOLUCION_TEXTO>>", res_text[:8000]) \
                .replace("<<LISTA_PRECEDENTES_CON_CITAS>>", precedentes_str)
    return prompt



def call_llm(prompt: str, api_key: str):
    openai.api_key= api_key

    resp = openai.ChatCompletion.create(
        model=MODEL_LLM,
        messages=[
            {"role":"system","content":"Eres un asistente experto en derecho administrativo chileno y compras públicas. Devuelve SIEMPRE JSON válido."},
            {"role":"user","content":prompt}
        ],
        temperature=0.2,
        max_tokens=1600
    )
    return resp.choices[0].message.content.strip()





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="preprocesamiento/actas_limpias.jsonl", help="JSONL con las actas (campos: archivo, texto_limpio)")
    ap.add_argument("--match", help="Subcadena a buscar en 'archivo' (case-insensitive)")
    ap.add_argument("--doc_id", help="Si tu JSONL tuviera 'doc_id' y quieres filtrar por él")
    ap.add_argument("--index", default="modulos/embedding-corpus/gold.index", help="Índice FAISS del gold")
    ap.add_argument("--mapping", default="modulos/embedding-corpus/gold_mapping.jsonl", help="Mapping del gold")
    ap.add_argument("--tpl", default="modulos/llm/prompt_template.txt")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--index_pos", type=int, help="si hay varias coincidencias, escoger por índice (0-based)")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Falta OPENAI_API_KEY (define en .env)")

    # cargar JSONL
    records = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    if not records:
        raise SystemExit("JSONL vacío o ilegible.")

    # filtrar por doc_id (si existiera) o por match en 'archivo'
    cands = records
    if args.doc_id:
        cands = [r for r in cands if str(r.get("doc_id","")) == args.doc_id]
    if args.match:
        m = args.match.lower()
        tmp = []
        for r in cands:
            name = str(r.get("archivo",""))
            if args.match in name:
                tmp.append(r)
        cands = tmp

    if not cands:
        raise SystemExit("No se encontraron coincidencias (usa --match o --doc_id).")

    rec = cands[args.index_pos] if (args.index_pos is not None and 0 <= args.index_pos < len(cands)) else cands[0]

    # texto directo desde el JSONL
    res_text = pick_text(rec)
    if not res_text.strip():
        raise SystemExit("El registro no tiene 'texto_limpio' con contenido.")

    # cargar índice GOLD
    index = faiss.read_index(args.index)
    gold_map = load_mapping(args.mapping)

    # embedding + búsqueda en GOLD
    emb_model = SentenceTransformer(MODEL_EMB)
    q = emb_model.encode([res_text], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q, args.k)
    precedentes = [gold_map[i] for i in I[0]]

    # prompt y LLM
    prompt = build_prompt(Path(args.tpl), res_text, precedentes)
    raw = call_llm(prompt, api_key)

    # parseo JSON
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"error":"No se pudo parsear JSON", "raw_output": raw}

    # salidas
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = normalize_base_name(rec)
    json_path = outdir / f"{base}.json"
    md_path = outdir / f"{base}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Informe de Riesgos – {base}\n\n")
        if "riesgos" in parsed:
            for r in parsed["riesgos"]:
                f.write(f"## {r.get('tipo','(sin tipo)')}\n")
                f.write(f"- Evidencia: {r.get('evidencia_resolucion')}\n")
                f.write(f"- Precedentes: {r.get('precedentes')}\n")
                f.write(f"- Recomendación: {r.get('recomendacion')}\n\n")
        else:
            f.write("No se detectaron riesgos o hubo error.\n")

    print(f"[OK] Guardado:\n- {json_path}\n- {md_path}")

if __name__ == "__main__":
    main()
