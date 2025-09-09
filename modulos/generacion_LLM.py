import openai
import json

# Establece tu clave API de OpenAI
openai.api_key = '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'  # Sustituye con tu clave real




# Cargar el mapeo de FAISS a documentos desde el archivo JSON
try:
    with open('modulos/faiss_documents.json', 'r', encoding='utf-8') as infile:
        faiss_documents = json.load(infile)
    print("Archivo 'faiss_documents.json' cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el archivo 'faiss_documents.json': {e}")

# Función para generar una respuesta con GPT-3.5
def generate_response_with_gpt35(query, faiss_results):
    # Formatear los textos de FAISS recuperados
    context = " ".join([result['texto'] for result in faiss_results])
    
    # Llamar a la API de GPT-3.5 usando el endpoint de chat
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Usamos GPT-3.5 Turbo
        messages=[
            {"role": "system", "content": "Eres un asistente útil experto en licitaciones publicas, ley chilena y resoluciones de adjudicación."},
            {"role": "user", "content": query},
            {"role": "system", "content": context}
        ],
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message['content'].strip()



# Los índices devueltos por la consulta semántica de FAISS
faiss_indices = [360]  # Los índices devueltos por FAISS
faiss_results = []  # Aquí almacenamos los resultados recuperados de los índices FAISS

# Verificar el número total de chunks generados
total_chunks = len(faiss_documents)  # Los chunks deben estar en faiss_documents.json
#print(f"Total de chunks generados: {total_chunks}")


# Buscar los resultados en FAISS y recuperar los documentos correspondientes
for idx in faiss_indices:
    if 0 <= idx < total_chunks:
        # Los índices de FAISS están dentro del rango de chunks generados
        document_info = faiss_documents.get(str(idx), {"documento": "No encontrado", "texto": "No encontrado"})
        faiss_results.append(document_info)  # Guardamos el documento y el texto limpio de FAISS
        print(document_info)
    else:
        print(f"Índice {idx} fuera del rango de los chunks generados.")




# Usar GPT-3.5 para generar una respuesta enriquecida con la información recuperada
query = "Falta informacion importante en este documento de decreto que debiesen de tener en cuenta resoluciones de adjudicación?"  # La consulta que se va a enviar a GPT-3.5
response = generate_response_with_gpt35(query, faiss_results)




# Mostrar la respuesta generada
print(f"Respuesta generada por GPT-3.5:\n{response}")
