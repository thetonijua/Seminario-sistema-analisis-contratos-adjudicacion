import argparse, json, os, re
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai
import random
import unicodedata
import hashlib


# ===== config =====
MODEL_EMB = "paraphrase-mpnet-base-v2"
MODEL_LLM = "gpt-4.1"
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

def normalize_labels(parsed: dict) -> dict:
    if not isinstance(parsed, dict) or "riesgos" not in parsed:
        return parsed
    map_labels = {
        "MS.Bases": "MR.Bases",
        "MR.Ley19.886": "MR.Ley19886",
        "MR.Ley 19.886": "MR.Ley19886",
        "MR. Ley19886": "MR.Ley19886",
        "MR.Ley 19886": "MR.Ley19886",
        "MR.19886": "MR.Ley19886",
    }
    catalog = set([
        "MS.Vistos","MS.Considerando","MR.Bases","MR.Ley19886","MR.Ley18695",
        "ID.Incorrecto","FMT.TituloID","VAL.Monto","DESC.Servicio"
    ])
    nuevos = []
    for r in parsed.get("riesgos", []):
        t = r.get("tipo")
        if t in map_labels:
            t = map_labels[t]
            r["tipo"] = t
        if t in catalog:
            nuevos.append(r)
        # si sigue fuera de catálogo, lo descartamos
    parsed["riesgos"] = nuevos
    return parsed


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

def mk_doc_id_from_filename(fname: str) -> str:
    """doc_id estable a partir del nombre de archivo (sin extensión)."""
    if not fname:
        return "doc_" + hashlib.md5(b"").hexdigest()[:8]

    # Normaliza a NFC para consistencia, conserva acentos y ‘°/º’
    s = unicodedata.normalize("NFC", fname)
    s = s.replace(" ", "_")

    # Permitidos: letras/dígitos/_-.() y caracteres latinos con tilde, además de ° y º y ‘N’
    s = re.sub(r"[^A-Za-z0-9_\-\.()ÁÉÍÓÚáéíóúÑñÜü°ºN°]", "", s)

    # Evita doc_id vacío
    if not s:
        s = "doc_" + hashlib.md5(fname.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return s

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
    ap.add_argument("--tpl", default="modulos/llm/prompt_templatev2.txt")
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
    ap.add_argument("--allow_on_critic_fail", action="store_true",help="Permite guardar en el pool aunque el Crítico marque ok=false")




    return ap.parse_args()














# ---------- MAIN ----------
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Falta OPENAI_API_KEY (define en .env)")

    args = parse_args()
    # planner prompt (una vez)
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

    procesados_ok = 0

    for idx, rec in enumerate(cands, start=1):
        print("\n===============================")
        print(f"[INFO] Documento {idx}/{len(cands)}: {rec.get('archivo')}")
        print("===============================\n")

        # Reiniciar control de acciones por documento
        last_action = None
        same_action_streak = 0

        # k/rag por documento (desde CLI)
        k_current = max(0, args.k)
        rag_enabled = (k_current > 0)

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
        critic = {}  # para que exista aunque falle antes

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

            action = (plan.get("action") or "").upper()
            args_a = plan.get("args", {}) or {}

            # Intercepción: si planner pide RETRIEVE_GOLD pero RAG está OFF, forzar eval
            if action == "RETRIEVE_GOLD" and not rag_enabled:
                print("[PLAN] Planner pidió RETRIEVE_GOLD pero RAG está desactivado → forzando EVAL_RISKS.")
                action, args_a = "EVAL_RISKS", {}

            # Failsafe: si repite acción demasiadas veces, forzamos EVAL_RISKS
            if action == last_action:
                same_action_streak += 1
            else:
                same_action_streak = 0
            last_action = action
            if same_action_streak >= 2:
                print("[PLAN] Acción repetida demasiadas veces → forzando EVAL_RISKS.")
                action, args_a = "EVAL_RISKS", {}

            # si ya hay precedentes y el planner vuelve a pedir RETRIEVE_GOLD, evalúar ya
            if action == "RETRIEVE_GOLD" and scratch.get("precedentes") and rag_enabled:
                print("[PLAN] Ya hay precedentes → saltando a EVAL_RISKS.")
                action, args_a = "EVAL_RISKS", {}


            # === ACCIONES ===
            if action == "RETRIEVE_GOLD":
                # respetar rag_enabled y k_current
                k_req = int(args_a.get("k", k_current))
                # alinear con runtime
                if k_req != k_current:
                    print(f"[ACT] Ajustando k del planner de {k_req} → {k_current} (valor runtime).")
                    k_req = k_current

                if not rag_enabled or k_req <= 0:
                    print("[ACT] RETRIEVE_GOLD → k=0 / RAG OFF (sin recuperación).")
                    scratch["precedentes"] = []
                    k_current = 0
                    scratch["k_current"] = k_current
                    continue

                print(f"[ACT] RETRIEVE_GOLD k={k_req} → buscando precedentes...")
                q = emb_model.encode([res_text], convert_to_numpy=True, normalize_embeddings=True)
                D, I = index.search(q, k_req)
                precs = [gold_map[i] for i in I[0]]
                scratch["precedentes"] = precs
                print("[ACT] Precedentes recuperados:", len(precs))
                k_current = k_req
                scratch["k_current"] = k_current
                continue

            if action == "ADJUST_K":
                # subir/bajar k respetando límites razonables
                k_current = int(args_a.get("k", min(10, max(0, k_current+3))))
                rag_enabled = (k_current > 0)
                scratch["k_current"] = k_current
                scratch["rag_enabled"] = rag_enabled
                print("[ACT] ADJUST_K → k_current:", k_current, "| rag_enabled:", rag_enabled)
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
                    # Un intento de reparación
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

        try:
            parsed = json.loads(final_json)
            parsed = normalize_labels(parsed)
            parsed = apply_consistency_guards(res_text, parsed, verbose=True)
        except json.JSONDecodeError:
            print("[ERROR] Salida no-JSON; se omite este documento.")
            continue

        # ---------- GUARDADO EN POOL JSONL ----------
        if args.pool_jsonl:
            orig_filename = rec.get("archivo") or rec.get("filename")
            if not orig_filename:
                orig_filename = "desconocido.pdf"

            # Normaliza a forward slashes
            filepath = str(Path(args.pool_data_root) / orig_filename).replace("\\", "/")

            doc_id_in = rec.get("doc_id")
            if not doc_id_in:
                base_name = Path(orig_filename).stem
                doc_id_in = mk_doc_id_from_filename(base_name)
            else:
                doc_id_in = str(doc_id_in).replace(".pdf", "").replace(".doc", "").strip()
            pool_item = {
                "doc_id": doc_id_in ,
                "filename": rec.get("filename") or orig_filename,   # conserva símbolos
                "filepath": filepath,                                # sólo slashes
                "riesgos": collect_risk_types(parsed),
                "evidencias": collect_evidencias(parsed),
                "nota_curador": build_nota_curador(parsed),
            }

            critic_ok_flag = bool(critic.get("ok", False))
            if not critic_ok_flag and not getattr(args, "allow_on_critic_fail", False):
                print("[POOL] Omitido: el Crítico no validó (ok=false). Usa --allow_on_critic_fail para forzar.")
                continue

            append_to_pool(Path(args.pool_jsonl), pool_item, dedup=args.pool_dedup)

        procesados_ok += 1

    print(f"\n[DONE] Documentos procesados: {procesados_ok}/{len(cands)}")


if __name__ == "__main__":
    main()
