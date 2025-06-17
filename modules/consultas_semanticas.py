import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Cargar el índice FAISS desde el archivo
def load_faiss_index(index_file='embeddings.index'):
    index = faiss.read_index(index_file)
    return index

# Función para realizar una consulta semántica
def semantic_search(query, index, model, k=5):
    # Convertir la consulta a embedding
    query_embedding = model.encode([query])
    
    # Realizar la búsqueda en FAISS para encontrar los k chunks más similares
    D, I = index.search(np.array(query_embedding).astype('float32'), k)  # D = distancias, I = índices de los documentos más cercanos
    
    return D, I

# Cargar el modelo de Sentence-Transformers
model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Cargar el índice FAISS desde el archivo
index = load_faiss_index('embeddings.index')

# Hacer una consulta
query = "Municipalidad de Rengo"

# Realizar la búsqueda semántica
distances, indices = semantic_search(query, index, model)

# Mostrar los resultados
print("Resultados más similares:")
for i in range(len(indices[0])):
    print(f"Resultado {i+1}:")
    print(f"Distancia: {distances[0][i]}")
    print(f"Índice: {indices[0][i]}")
    print()
