import json, argparse
from collections import defaultdict


#Solo para efectos de evaluación, se usa el mismo catálogo que en el análisis

CATALOGO = {
    "MS.Vistos","MS.Considerando","MR.Bases","MR.Ley19886","MR.Ley18695",
    "ID.Incorrecto","FMT.TituloID","VAL.Monto","DESC.Servicio"
}

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

def as_set(labels):
    return {x for x in labels if x in CATALOGO}

def index_by_doc(items):
    m = {}
    for x in items:
        # formato esperado: {"doc_id","riesgos":[...]}
        m[x["filename"]] = as_set(x.get("riesgos", []))
    return m

def prf1(tp, fp, fn):
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
    return p,r,f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default= "experimentos/splits/gold_test.jsonl")
    ap.add_argument("--pred",default="experimentos/runs/pool_ragON_k5_v3.jsonl")
    ap.add_argument("--out", default="experimentos/reports/metrics_ragON_k5_v3.json")
    args = ap.parse_args()

    gold = index_by_doc(load_jsonl(args.gold))
    pred = index_by_doc(load_jsonl(args.pred))

    labels = sorted(CATALOGO)
    per_label = {l: {"tp":0,"fp":0,"fn":0} for l in labels}

    # Alinea por doc_id presente en gold
    docs = sorted(gold.keys())
    missing = [d for d in docs if d not in pred]
    if missing:
        print(f"[WARN] {len(missing)} docs del gold no aparecen en predicciones.")

    for d in docs:
        g = gold[d]
        p = pred.get(d, set())

        for l in labels:
            g_has = (l in g)
            p_has = (l in p)
            if g_has and p_has:
                per_label[l]["tp"] += 1
            elif not g_has and p_has:
                per_label[l]["fp"] += 1
            elif g_has and not p_has:
                per_label[l]["fn"] += 1
            # else: true negative (no cuenta)

    # micro
    micro_tp = sum(v["tp"] for v in per_label.values())
    micro_fp = sum(v["fp"] for v in per_label.values())
    micro_fn = sum(v["fn"] for v in per_label.values())
    micro = prf1(micro_tp, micro_fp, micro_fn)

    # macro
    macro_f1s = []
    rows = []
    for l in labels:
        p,r,f1 = prf1(**per_label[l])
        rows.append((l, per_label[l]["tp"], per_label[l]["fp"], per_label[l]["fn"], p, r, f1))
        macro_f1s.append(f1)
    macro_f1 = sum(macro_f1s)/len(macro_f1s) if macro_f1s else 0.0

    # print resumen
    print("\n== Per-label ==")
    for l,tp,fp,fn,p,r,f1 in rows:
        print(f"{l:15s}  TP={tp:3d} FP={fp:3d} FN={fn:3d}  P={p:.3f} R={r:.3f} F1={f1:.3f}")

    print("\n== Micro ==")
    print(f"TP={micro_tp} FP={micro_fp} FN={micro_fn}  P={micro[0]:.3f} R={micro[1]:.3f} F1={micro[2]:.3f}")

    print("\n== Macro-F1 ==")
    print(f"Macro-F1={macro_f1:.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "per_label":[{"label":l,"tp":tp,"fp":fp,"fn":fn,"precision":p,"recall":r,"f1":f1}
                             for l,tp,fp,fn,p,r,f1 in rows],
                "micro":{"precision":micro[0],"recall":micro[1],"f1":micro[2]},
                "macro_f1": macro_f1
            }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
