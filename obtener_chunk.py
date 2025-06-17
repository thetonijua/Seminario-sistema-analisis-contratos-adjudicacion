import json

# Cargar el mapeo de FAISS a documentos desde el archivo JSON
try:
    with open('faiss_documents.json', 'r', encoding='utf-8') as infile:
        faiss_documents = json.load(infile)
    print("Archivo 'faiss_documents.json' cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el archivo 'faiss_documents.json': {e}")

# Los índices devueltos por la consulta semántica de FAISS
faiss_indices = [916, 360, 511, 398, 390]  # Los índices devueltos por FAISS

# Verificar el número total de chunks generados
total_chunks = len(faiss_documents)  # Los chunks deben estar en faiss_documents.json
print(f"Total de chunks generados: {total_chunks}")

# Mostrar los resultados con el mapeo de los índices FAISS a los documentos originales
for idx in faiss_indices:
    if 0 <= idx < total_chunks:
        # Los índices de FAISS están dentro del rango de chunks generados
        document = faiss_documents.get(str(idx), "Documento no encontrado")  # Usamos str(idx) porque el JSON tiene claves de tipo string
        print(f"Índice de FAISS: {idx}")
        print(f"Archivo: {document}")
        print("----")
    else:
        print(f"Índice {idx} fuera del rango de los chunks generados.")
