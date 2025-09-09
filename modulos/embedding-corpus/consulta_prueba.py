# Ejemplo rápido de consulta
from sentence_transformers import SentenceTransformer
import faiss
import json

# Cargar
index = faiss.read_index("modulos/embedding-corpus/gold.index")
mapping = [json.loads(l) for l in open("modulos/embedding-corpus/gold_mapping.jsonl", encoding="utf-8")]

model = SentenceTransformer("paraphrase-mpnet-base-v2")

query_text = "Falta sección Considerando y no se cita la Ley 19.886"
q = model.encode([query_text], convert_to_numpy=True)

D, I = index.search(q, 5)
for rank, idx in enumerate(I[0], 1):
    print(rank, mapping[idx]["doc_id"], mapping[idx]["riesgos"], mapping[idx]["nota_curador"])
