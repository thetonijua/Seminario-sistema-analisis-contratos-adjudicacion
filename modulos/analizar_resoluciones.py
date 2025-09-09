import argparse, json, os, re
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai

# ===== config =====
MODEL_EMB = "paraphrase-mpnet-base-v2"
MODEL_LLM = "gpt-3.5-turbo"
CATALOGO = [
    "MS.Vistos",
    "MS.Considerando",
    "MR.Bases",
    "MR.Ley19886",
    "MR.Ley18695",
    "ID.Incorrecto",
    "FMT.TituloID",
    "VAL.Monto",
    "DESC.Servicio"
]
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



def run_planner(prompt_sys: str, scratch: dict, api_key: str):

    openai.api_key= api_key

    msg = [
        {"role":"system","content": prompt_sys},
        {"role":"user","content": json.dumps({
            "doc_summary":"Resumen corto del documento (opcional)",
            "known_catalog": CATALOGO,
            "observations": scratch  # lo que ya tenemos (hits regex, ventanas base, etc.)
        }, ensure_ascii=False)}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=msg,
        temperature=0.0,
        max_tokens=400
    )
    return resp.choices[0].message.content.strip()



def call_critic(doc_block: str, analyst_json: str, api_key: str) -> str:
    """
    Llama al agente Crítico (LLM) que valida:
    - JSON válido
    - Etiquetas dentro del catálogo
    - Citas literales presentes en DOCUMENTO_BASE y que calzan con loc char[a:b]
    Devuelve SIEMPRE el texto de salida del LLM (que debería ser JSON).
    """
    openai.api_key= api_key

    # Lee el prompt del crítico
    critic_path = Path("modulos/llm/plan_prompt.txt")
    if not critic_path.exists():
        # Fallback por si falta el archivo (evita crash)
        critic_tpl = (
            'ROL: Eres un auditor de calidad. Valida el JSON del Analista.\n'
            'CATALOGO_PERMITIDO: ["MS.Vistos","MS.Considerando","MR.Bases","MR.Ley19886","MR.Ley18695",'
            '"ID.Incorrecto","FMT.TituloID","VAL.Monto","DESC.Servicio"]\n'
            'Requisitos: 1) JSON válido, 2) tipo en catálogo, 3) cada evidencia literal existe en DOCUMENTO_BASE y coincide con loc.\n'
            'Salida: {"ok": true} o {"ok": false, "errores":[{"campo":"...","motivo":"...","sugerencia":"..."}]}\n'
        )
    else:
        critic_tpl = critic_path.read_text(encoding="utf-8")

    # Payload para el crítico
    user_payload = f"DOCUMENTO_BASE:\n{doc_block}\n\nPROPUESTA_ANALISTA:\n{analyst_json}"

    # Llamada al LLM
    resp = openai.ChatCompletion.create(
        model=MODEL_LLM,
        messages=[
            {"role": "system", "content": critic_tpl},
            {"role": "user", "content": user_payload}
        ],
        temperature=0.0,
        max_tokens=800
    )
    return resp.choices[0].message.content.strip()


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="preprocesamiento/actas_limpias.jsonl", help="JSONL con las actas (campos: archivo, texto_limpio)")
    ap.add_argument("--match", help="Nombre EXACTO en 'archivo' (incluida extensión)")
    ap.add_argument("--doc_id", help="Si tu JSONL tuviera 'doc_id' y quieres filtrar por él")
    ap.add_argument("--index", default="modulos/embedding-corpus/gold.index", help="Índice FAISS del gold")
    ap.add_argument("--mapping", default="modulos/embedding-corpus/gold_mapping.jsonl", help="Mapping del gold")
    ap.add_argument("--tpl", default="modulos/llm/prompt_template.txt")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--index_pos", type=int, help="si hay varias coincidencias, escoger por índice (0-based)")
    ap.add_argument("--dump_docblock", help="Ruta para guardar el DOCUMENTO_BASE que se envía al LLM (ventanas + pistas)")
    ap.add_argument("--dump_prompt", help="Ruta para guardar el prompt completo enviado al Analista")
    ap.add_argument("--fulltext", action="store_true", help="Pasa el texto completo al LLM en lugar de ventanas")

    return ap.parse_args()














# ---------- MAIN ----------
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Falta OPENAI_API_KEY (define en .env)")

    args = parse_args()

    print("[INFO] Cargando JSONL:", args.jsonl)

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
    print("[INFO] Registros total:", len(records))

    # filtrar por doc_id o match EXACTO en 'archivo'
    cands = records
    if args.doc_id:
        cands = [r for r in cands if str(r.get("doc_id","")) == args.doc_id]
    if args.match:
        tmp = []
        for r in cands:
            name = str(r.get("archivo",""))
            if name == args.match:  # exacto
                tmp.append(r)
        cands = tmp

    print("[INFO] Coincidencias por filtro:", len(cands))
    if not cands:
        raise SystemExit("No se encontraron coincidencias (usa --match exacto o --doc_id).")

    rec = cands[args.index_pos] if (args.index_pos is not None and 0 <= args.index_pos < len(cands)) else cands[0]
    print("[INFO] Seleccionado:", rec.get("archivo"))

    # texto
    res_text = pick_text(rec)
    if not res_text.strip():
        raise SystemExit("El registro no tiene 'texto_limpio' con contenido.")
    print("[INFO] Largo del texto:", len(res_text), "chars")

    # DOCUMENTO_BASE (full-text siempre; mantenemos --fulltext por compatibilidad)
    base_doc_block = "[DOCUMENTO_COMPLETO]\n" + res_text

    # Cargar FAISS GOLD
    index = faiss.read_index(args.index)
    gold_map = load_mapping(args.mapping)
    print("[INFO] FAISS index cargado:", args.index, "Mapping:", args.mapping)

    # Estado del planner (simplificado: sin hits ni ventanas)
    scratch = {
        "doc_id": rec.get("archivo"),
        "precedentes": []
    }

    planner_sys = Path("modulos/llm/planner_prompt.txt").read_text(encoding="utf-8")
    max_steps, step = 4, 0
    final_json = None
    k_current = args.k

    # BUCLE AGENTIC 
    while step < max_steps:
        step += 1
        print(f"[PLAN] step={step} solicitando plan...")
        plan_raw = run_planner(planner_sys, scratch, api_key)  # se le pasa el planner prompt
        try:
            plan = json.loads(plan_raw)
        except Exception:
            plan = {"thought":"fallback eval","action":"EVAL_RISKS","args":{}}
        print("[PLAN] Respuesta planner:", plan)

        action = plan.get("action")
        args_a = plan.get("args", {})

        if action == "SCAN_DOC":
            # ya no usamos regex/ventanas; tratamos SCAN_DOC como no-op
            print("[ACT] SCAN_DOC → ignorado (modo full-text).")
            continue

        if action == "RETRIEVE_GOLD":
            k_req = int(args_a.get("k", k_current))
            print(f"[ACT] RETRIEVE_GOLD k={k_req} → buscando precedentes...")
            emb_model = SentenceTransformer(MODEL_EMB)
            q = emb_model.encode([res_text], convert_to_numpy=True, normalize_embeddings=True)
            D, I = index.search(q, k_req)
            precs = [gold_map[i] for i in I[0]]
            scratch["precedentes"] = precs
            print("[ACT] Precedentes recuperados:", len(precs))
            k_current = k_req
            continue

        if action == "ADJUST_K":
            k_current = int(args_a.get("k", min(10, max(5, k_current+3))))
            print("[ACT] ADJUST_K → k_current:", k_current)
            continue

        if action == "EVAL_RISKS":
            print("[ACT] EVAL_RISKS → enviando al analista (LLM)")
            precedentes = scratch.get("precedentes", [])
            prompt = build_prompt(Path(args.tpl), base_doc_block, precedentes)
            raw = call_llm(prompt, api_key)
            print("[LLM] Analista devolvió", len(raw), "chars")

            # crítico sobre el TEXTO COMPLETO (full-text)
            critic_out = call_critic(
                doc_block=res_text,
                analyst_json=raw,
                api_key=api_key
            )
            try:
                critic = json.loads(critic_out)
            except json.JSONDecodeError:
                critic = {"ok": False, "errores":[{"motivo":"Crítico devolvió texto no JSON","sugerencia":"revisar prompt crítico"}]}
            print("[CRITIC] Salida crítico:", critic)

            if critic.get("ok", False):
                final_json = raw
                break
            else:
                # reparación única
                feedback = json.dumps(critic, ensure_ascii=False)
                repair_prompt = prompt + "\n\n[REVISION_DEL_AUDITOR]\n" + feedback + \
                                "\nCorrige tu JSON: ajusta/añade citas literales o elimina riesgos sin evidencia. Devuelve SOLO JSON."
                raw2 = call_llm(repair_prompt, api_key)

                critic_out2 = call_critic(
                    doc_block=res_text,
                    analyst_json=raw2,
                    api_key=api_key
                )
                ok2 = False
                try:
                    ok2 = json.loads(critic_out2).get("ok", False)
                except json.JSONDecodeError:
                    ok2 = False

                final_json = raw2 if ok2 else raw
                break

        if action == "FINALIZE":
            print("[ACT] FINALIZE solicitado por planner.")
            break

    # Fallback si no hubo dictamen
    if final_json is None:
        print("[WARN] Planner no produjo dictamen; forzando evaluación directa.")
        precedentes = scratch.get("precedentes", [])
        prompt = build_prompt(Path(args.tpl), base_doc_block, precedentes)
        final_json = call_llm(prompt, api_key)

    # Parseo final
    try:
        parsed = json.loads(final_json)
    except json.JSONDecodeError:
        parsed = {"error":"No se pudo parsear JSON", "raw_output": final_json}

    # Guardado
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = normalize_base_name(rec)  # si quieres exacto exacto: base = rec.get("archivo","acta")
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

    print(f"[DONE] Guardado:\n- {json_path}\n- {md_path}")

if __name__ == "__main__":
    main()
