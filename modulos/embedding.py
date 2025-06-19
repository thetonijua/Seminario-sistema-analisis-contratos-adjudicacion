import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Función para generar embeddings
def generate_embeddings(texts):
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

# Función para almacenar los embeddings en FAISS
def store_embeddings_in_faiss(embeddings, index_file='embeddings.index'):
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 para la distancia euclidiana
    index.add(embeddings)
    faiss.write_index(index, index_file)
    print(f"Índice FAISS guardado en: {index_file}")
    return index

# Leer el archivo JSONL y obtener los textos para generar embeddings
input_file = 'actas_limpias.jsonl'
texts = []
documents = []

# Leer y procesar el archivo JSONL
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        acta = json.loads(line)
        if 'texto_limpio' in acta:
            # Guardamos los textos completos sin dividirlos
            texts.append(acta['texto_limpio'])
            documents.append(acta['archivo'])  # Guardar el archivo original para cada documento

# Generar los embeddings para todos los textos completos
embeddings = generate_embeddings(texts)

# Almacenar los embeddings en FAISS
index = store_embeddings_in_faiss(embeddings)

# Guardar el mapeo entre los índices de FAISS, los documentos originales y el texto limpio de cada chunk
faiss_documents = {i: {'documento': doc, 'texto': texts[i]} for i, doc in enumerate(documents)}

# Guardar el mapeo en un archivo JSON para su uso posterior
with open('faiss_documents.json', 'w', encoding='utf-8') as outfile:
    json.dump(faiss_documents, outfile, ensure_ascii=False, indent=4)

print("Generación de embeddings, almacenamiento en FAISS y mapeo guardado en 'faiss_documents.json' completado.")
