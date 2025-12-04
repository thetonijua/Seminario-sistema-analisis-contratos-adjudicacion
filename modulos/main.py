import argparse, json, os, re
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai
import random


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
        temperature=0.0,
        max_tokens=2000
    )
    return resp.choices[0].message.content.strip()



def run_planner(prompt_sys: str, scratch: dict, api_key: str):
    openai.api_key = api_key
    k_cur = scratch.get("k_current", 5)
    rag_en = scratch.get("rag_enabled", True)

    sys_dyn = (
        prompt_sys
        + f"\n\n[CONSTRAINTES RUNTIME]\n"
          f"- k_current={k_cur}\n"
          f"- rag_enabled={'true' if rag_en else 'false'}\n"
          f"- Si rag_enabled=false o k_current==0: NO uses RETRIEVE_GOLD.\n"
          f"- Si usas RETRIEVE_GOLD, debes devolver args.k={k_cur} EXACTO (no uses 5 por defecto).\n"
    )

    msg = [
        {"role":"system","content": sys_dyn},
        {"role":"user","content": json.dumps({
            "doc_summary":"(opcional)",
            "known_catalog": CATALOGO,
            "observations": scratch
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
    critic_path = Path("modulos/llm/critic_prompt.txt")
    if not critic_path.exists():
        # Fallback mínimo si falta archivo
        critic_tpl = (
            'ROL: Eres un auditor de calidad. Valida el JSON del Analista.\n'
            'CATALOGO_PERMITIDO: ["MS.Vistos","MS.Considerando","MR.Bases","MR.Ley19886","MR.Ley18695",'
            '"ID.Incorrecto","FMT.TituloID","VAL.Monto","DESC.Servicio"]\n'
            'Reglas: 1) JSON válido, 2) tipo en catálogo, 3) cita literal debe existir.\n'
            'Si no hay riesgos, permite {"ok": true, "riesgos": [], "mensaje":"Sin riesgo detectado"}.\n'
            'Salida: {"ok": true} o {"ok": false, "errores":[...]}.\n'
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
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()

def mk_doc_id_from_filename(filename: str) -> str:
    """doc_id estable a partir del nombre de archivo (sin extensión)."""
    base = Path(filename).stem if filename else "doc"
    base = re.sub(r"[^\w\-]+", "_", base, flags=re.UNICODE)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "doc"

def collect_risk_types(parsed: dict) -> list:
    tipos = []
    for r in parsed.get("riesgos", []) or []:
        t = (r.get("tipo") or "").strip()
        if t:
            tipos.append(t)
    return sorted(set(tipos))

def collect_evidencias(parsed: dict) -> list:
    evs = []
    for r in parsed.get("riesgos", []) or []:
        raw = r.get("evidencia_resolucion") or []
        for ev in raw:
            if isinstance(ev, dict):
                cita = str(ev.get("cita", "") or ev.get("texto", "")).strip()
                loc  = str(ev.get("loc", "n/a")).strip() or "n/a"
                secc = str(ev.get("seccion", "Global")).strip() or "Global"
                if cita:
                    evs.append({"texto": cita, "seccion": secc, "loc": loc})
            else:
                # cadena simple
                cita = str(ev).strip()
                if cita:
                    evs.append({"texto": cita, "seccion": "Global", "loc": "n/a"})
    return evs

def build_nota_curador(parsed: dict) -> str:
    # Prioriza 'resumen'; si no hay, usa primera 'recomendacion'; sino, síntesis de tipos
    resumen = (parsed.get("resumen") or "").strip()
    if resumen:
        return resumen
    for r in parsed.get("riesgos", []) or []:
        rec = (r.get("recomendacion") or "").strip()
        if rec:
            return rec
    tipos = collect_risk_types(parsed)
    return ("Detectado(s): " + ", ".join(tipos)) if tipos else "Revisión automática sin hallazgos destacables."

def append_to_pool(pool_path: Path, item: dict, dedup: bool):
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    if dedup and pool_path.exists():
        # carga IDs existentes para evitar duplicados
        seen = set()
        with pool_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                try:
                    obj = json.loads(line)
                    did = obj.get("doc_id")
                    if did: seen.add(did)
                except Exception:
                    pass
        if item.get("doc_id") in seen:
            print("[POOL] Omitido (duplicado por doc_id):", item.get("doc_id"))
            return
    with pool_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("[POOL] Agregado en", str(pool_path))



def _has(text: str, pat: str) -> bool:
    return re.search(pat, text, flags=re.IGNORECASE) is not None

def apply_consistency_guards(res_text: str, parsed: dict, verbose: bool = True) -> dict:
    """
    Anti-falsos-positivos: elimina riesgos incoherentes con el propio documento.
    Si no queda ninguno, marca 'Sin riesgo detectado'.
    """
    if not isinstance(parsed, dict):
        return parsed

    riesgos = parsed.get("riesgos") or []
    if not isinstance(riesgos, list):
        return parsed

    text = res_text  # usamos IGNORECASE en los patrones

    # equivalencias típicas
    pat_ley_19886 = r"ley\s*(n|nº|n°)?\s*19[\s\.\-]?886"
    pat_ley_18695 = r"(ley\s*(n|nº|n°)?\s*18[\s\.\-]?695)|(org[aá]nica\s+constitucional\s+de\s+municipalidades)"
    pat_vistos    = r"\b(visto?s?|antecedentes)\b"
    pat_cons      = r"\b(considerando?s?|motivaci[oó]n)\b"
    pat_monto     = r"(\$|\bpesos\b|\bclp\b)\s*\d[\d\.\s]*"
    pat_id_ok     = r"\b\d{3,4}-\d{1,4}-(L|LE|LP)\d{1,3}\b"
    pat_bases     = r"bases\s+(administrativas|t[eé]cnicas|de\s+licitaci[oó]n)"

    keep, dropped = [], []

    for r in riesgos:
        tipo = (r.get("tipo") or "").strip()

        if tipo == "MR.Ley19886" and _has(text, pat_ley_19886):
            dropped.append("MR.Ley19886 removido: el documento SÍ menciona Ley 19.886.")
            continue
        if tipo == "MR.Ley18695" and _has(text, pat_ley_18695):
            dropped.append("MR.Ley18695 removido: el documento SÍ menciona Ley 18.695/OCM.")
            continue
        if tipo == "MS.Vistos" and _has(text, pat_vistos):
            dropped.append("MS.Vistos removido: hay 'Vistos'/'Antecedentes'.")
            continue
        if tipo == "MS.Considerando" and _has(text, pat_cons):
            dropped.append("MS.Considerando removido: hay 'Considerando(s)'/'Motivación'.")
            continue
        if tipo == "VAL.Monto" and _has(text, pat_monto):
            dropped.append("VAL.Monto removido: hay monto en el texto.")
            continue
        if tipo == "ID.Incorrecto" and _has(text, pat_id_ok):
            dropped.append("ID.Incorrecto removido: hay un ID con formato válido.")
            continue
        if tipo == "MR.Bases" and _has(text, pat_bases):
            dropped.append("MR.Bases removido: se mencionan Bases (Administrativas/Técnicas/Licitación).")
            continue

        # FMT.TituloID y DESC.Servicio quedan sin filtro automático (son más contextuales)
        keep.append(r)

    parsed["riesgos"] = keep

    if verbose and dropped:
        print("[GUARDS] Remociones por coherencia:")
        for msg in dropped:
            print("  -", msg)

    if not keep:
        parsed["resumen"] = "Sin riesgo detectado."
        parsed["mensaje"] = "Sin riesgo detectado"

    return parsed




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
    ap.add_argument("--pool_jsonl", help="Si se entrega, se hace append de cada salida en formato metadata a este JSONL")
    ap.add_argument("--pool_data_root", default="data", help="Prefijo para filepath en el pool (por defecto 'data')")
    ap.add_argument("--pool_dedup", action="store_true", help="Evita duplicados por doc_id en el pool_jsonl")
    ap.add_argument("--random", action="store_true", help="Si se indica, selecciona al azar en vez de usar --match")
    ap.add_argument("--n_random", type=int, default=1, help="Número de documentos aleatorios a procesar si se usa --random")



    return ap.parse_args()














# ---------- MAIN ----------
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Falta OPENAI_API_KEY (define en .env)")

    args = parse_args()
    k_current = max(0, args.k)
    rag_enabled = (k_current > 0)

    # Seguimiento de acciones repetidas
    last_action = None
    same_action_streak = 0

    planner_sys = Path("modulos/llm/planner_prompt.txt").read_text(encoding="utf-8")



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
    elif args.match:
        tmp = []
        for r in cands:
            name = str(r.get("archivo",""))
            if name == args.match:  # exacto
                tmp.append(r)
        cands = tmp
    elif args.random:
        n = min(args.n_random, len(cands))
        cands = random.sample(cands, n)
        print(f"[INFO] Seleccionados {n} documentos aleatorios.")

    # Cargar FAISS GOLD y el modelo de embeddings UNA sola vez
    index = faiss.read_index(args.index)
    gold_map = load_mapping(args.mapping)
    emb_model = SentenceTransformer(MODEL_EMB)

    planner_sys = Path("modulos/llm/planner_prompt.txt").read_text(encoding="utf-8")


    procesados_ok = 0

    for idx, rec in enumerate(cands, start=1):
        print("\n===============================")
        print(f"[INFO] Documento {idx}/{len(cands)}: {rec.get('archivo')}")
        print("===============================\n")

        # --- TEXTO ---
        res_text = pick_text(rec)
        if not res_text or not res_text.strip():
            print("[WARN] Texto vacío; se omite este documento.")
            continue
        print("[INFO] Largo del texto:", len(res_text), "chars")

        # Documento base (full-text)
        base_doc_block = "[DOCUMENTO_COMPLETO]\n" + res_text

        # Estado del planner por documento
        scratch = {
            "doc_id": rec.get("archivo"),
            "precedentes": [],
            "k_current": k_current,
            "rag_enabled": rag_enabled
        }

        max_steps, step = 4, 0
        final_json = None


    # BUCLE AGENTIC 
        # ===== BUCLE AGENTIC =====
        while step < max_steps:
            step += 1

            print(f"[PLAN] step={step} solicitando plan...")

            plan_raw = run_planner(planner_sys, scratch, api_key)
            try:
                plan = json.loads(plan_raw)
            except Exception:
                plan = {"thought":"fallback eval","action":"EVAL_RISKS","args":{}}
            print("[PLAN] Respuesta planner:", plan)

            action = plan.get("action")
            args_a = plan.get("args", {})

             # ===  Intercepción antes del manejo de acciones ===
            if action == "RETRIEVE_GOLD" and not rag_enabled:
                print("[PLAN] Planner pidió RETRIEVE_GOLD pero RAG está desactivado → forzando EVAL_RISKS.")
                action, args_a = "EVAL_RISKS", {}

            # ===  Failsafe: si repite acción muchas veces, forzamos avance ===
            if action == last_action:
                same_action_streak += 1
            else:
                same_action_streak = 0
            last_action = action
            if same_action_streak >= 2:
                print("[PLAN] Acción repetida demasiadas veces → forzando EVAL_RISKS.")
                action, args_a = "EVAL_RISKS", {}


             # ===  ACCIONES ===
            if action == "RETRIEVE_GOLD" and not rag_enabled:
                
                k_req = max(0, int(args_a.get("k", k_current)))
                # fuerza coincidencia con k_current
                if k_req != k_current:
                    print(f"[ACT] Ajustando k del planner de {k_req} → {k_current} (valor runtime).")
                    k_req = k_current

                if k_req <= 0:
                    print("[ACT] RETRIEVE_GOLD → k=0 (sin recuperación).")
                    scratch["precedentes"] = []
                    k_current = 0
                    continue

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

                # Crítico sobre full-text
                critic_out = call_critic(doc_block=res_text, analyst_json=raw, api_key=api_key)
                try:
                    critic = json.loads(critic_out)
                except json.JSONDecodeError:
                    critic = {"ok": False, "errores":[{"motivo":"Crítico devolvió texto no JSON","sugerencia":"revisar prompt crítico"}]}
                print("[CRITIC] Salida crítico:", critic)

                if critic.get("ok", False):
                    final_json = raw
                    break
                else:
                    feedback = json.dumps(critic, ensure_ascii=False)
                    repair_prompt = prompt + "\n\n[REVISION_DEL_AUDITOR]\n" + feedback + \
                                    "\nCorrige tu JSON: ajusta/añade citas literales o elimina riesgos sin evidencia. Devuelve SOLO JSON."
                    raw2 = call_llm(repair_prompt, api_key)

                    critic_out2 = call_critic(doc_block=res_text, analyst_json=raw2, api_key=api_key)
                    try:
                        ok2 = json.loads(critic_out2).get("ok", False)
                    except json.JSONDecodeError:
                        ok2 = False

                    final_json = raw2 if ok2 else raw
                    break

            if action == "FINALIZE":
                print("[ACT] FINALIZE solicitado por planner.")
                break

        # Fallback si no hubo dictamen dentro del bucle
        if final_json is None:
            print("[WARN] Planner no produjo dictamen; forzando evaluación directa.")
            precedentes = scratch.get("precedentes", [])
            prompt = build_prompt(Path(args.tpl), base_doc_block, precedentes)
            final_json = call_llm(prompt, api_key)

        # Parseo final
        try:
            parsed = json.loads(final_json)
            # Cinturón anti-FP (remueve riesgos incoherentes con el propio documento)
            parsed = apply_consistency_guards(res_text, parsed, verbose=True)
        except json.JSONDecodeError:
            print("[ERROR] Salida no-JSON; se omite este documento.")
            continue

        # ---------- GUARDADO EN POOL JSONL ----------
        if args.pool_jsonl:
            orig_filename = rec.get("archivo")
            if orig_filename and "." not in Path(orig_filename).name:
                orig_filename = str(Path(orig_filename).with_suffix(".pdf"))
            if not orig_filename:
                orig_filename = normalize_base_name(rec) + ".pdf"

             # Normaliza a forward slashes
            filepath = str(Path(args.pool_data_root) / orig_filename).replace("\\", "/")

            pool_item = {
                "doc_id": mk_doc_id_from_filename(orig_filename),
                "filename": orig_filename,
                "filepath": filepath,
                "riesgos": collect_risk_types(parsed),
                "evidencias": collect_evidencias(parsed),
                "nota_curador": build_nota_curador(parsed)
            }


            # Asegura que critic_ok_flag exista (si no fue seteado antes, asume False)
            critic_ok_flag = bool(locals().get("critic", {}).get("ok", False))

            if not critic_ok_flag and not getattr(args, "allow_on_critic_fail", False):
                print("[POOL] Omitido: el Crítico no validó (ok=false). Usa --allow_on_critic_fail para forzar.")
                continue

            append_to_pool(Path(args.pool_jsonl), pool_item, dedup=args.pool_dedup)

        procesados_ok += 1

    print(f"\n[DONE] Documentos procesados: {procesados_ok}/{len(cands)}")


if __name__ == "__main__":
    main()
