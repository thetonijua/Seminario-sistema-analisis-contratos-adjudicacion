import openai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv
import os

# Cargar las variables del archivo .env
load_dotenv()


# Definir la clave API
openai.api_key = os.getenv("OPENAI_API_KEY")  

# Ruta base del archivo actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas absolutas a archivos de embeddings y texto
INDEX_PATH = os.path.join(BASE_DIR, "..", "modulos", "embeddings.index")
JSON_PATH = os.path.join(BASE_DIR, "..", "modulos", "faiss_documents.json")


# Función para cargar el índice FAISS
def load_faiss_index(index_file=INDEX_PATH):
    index = faiss.read_index(index_file)
    return index

# Función para realizar una consulta semántica
def semantic_search(query, index, model, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), k)
    return D, I

# Función para generar una respuesta con GPT
def generate_response_with_gpt(query, faiss_results, model):
    context = " ".join([result['texto'] for result in faiss_results])

    doc_nombres = [result['documento'] for result in faiss_results]
    doc_list = "\n".join(f"- {doc}" for doc in doc_nombres)

    prompt = (
        f"Los siguientes documentos han sido recuperados:\n{doc_list}\n\n"
        f"Basado en el contenido legal de estos documentos, responde a la siguiente consulta legal:\n"
        f"{query}\n\n"
        "Organiza tu respuesta por documento recuperado. Para cada uno, indica si cumple con las leyes actuales "
        "relativas a licitaciones en Chile y si están actualizadas, si falta alguna ley clave, y si se detectan posibles riesgos o ambigüedades. Como falta de informacion o claridad "
        "Menciona las leyes por su número y nombre cuando sea posible. Responde de forma clara, profesional y concisa."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un abogado experto en derecho administrativo y contrataciones públicas en Chile."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.4
    )

    return response.choices[0].message['content'].strip()


# Cargar el modelo de Sentence-Transformers
model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Cargar el índice FAISS
index = load_faiss_index('embeddings.index')

# Ciclo Iterativo para búsqueda y evaluación
def iterative_search_and_evaluation(query, max_iterations=3):
    # Realizar búsqueda inicial
    D, I = semantic_search(query, index, model)
    
    # Cargar los documentos de FAISS desde faiss_documents.json
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        faiss_documents = json.load(f)

    faiss_results = []  # Lista para almacenar los resultados de FAISS
    
    # Verificar que los índices de FAISS son válidos y agregar los documentos correspondientes
    for idx in I[0]:
        if 0 <= idx < len(faiss_documents):
            document_info = faiss_documents.get(str(idx), {"documento": "No encontrado", "texto": "No encontrado"})
            faiss_results.append(document_info)
            print(f"→ Documento recuperado: {document_info['documento']}")
        else:
            print(f"Índice {idx} fuera del rango de los chunks generados.")

    
    # Evaluar los resultados con GPT-4
    response = generate_response_with_gpt(query, faiss_results, model)
    
    print(f"Iteración 1: Respuesta generada por GPT-3.5:\n{response}\n")
    
    # Iterar si es necesario
    iteration = 1
    while iteration < max_iterations and "irregularidad" not in response.lower():
        print(f"Iteración {iteration + 1}: Refinando consulta...")
        query = "¿Se mencionan todas las leyes actuales y cláusulas importantes? Si falta alguna, ¿qué debería estar incluido?"
        
        # Realizar nueva búsqueda y evaluación
        D, I = semantic_search(query, index, model)
        faiss_results = []  # Limpiar resultados previos
        
        # Recargar los resultados de FAISS y volver a hacer el mapeo
        for idx in I[0]:
            if 0 <= idx < len(faiss_documents):  # Verificar si el índice está dentro del rango
                document_info = faiss_documents.get(str(idx), {"documento": "No encontrado", "texto": "No encontrado"})
                faiss_results.append(document_info)  # Guardar el documento y el texto limpio de FAISS
                #print(document_info)
                print(f"→ Documento recuperado: {document_info['documento']}")
            else:
                print(f"Índice {idx} fuera del rango de los chunks generados.")
        
        response = generate_response_with_gpt(query, faiss_results, model)
        print(f"Iteración {iteration + 1}: Respuesta generada por GPT-3.5:\n{response}\n")
        
        iteration += 1
        
    return response

# Ejemplo de consulta
query = "¿Este documento  si menciona las leyes actuales sobre licitaciones, tiene leyes desactualizadas o falta información que podria ser indicio de riesgo?"
response = iterative_search_and_evaluation(query)
print(f"Respuesta final del ciclo iterativo:\n{response}")
