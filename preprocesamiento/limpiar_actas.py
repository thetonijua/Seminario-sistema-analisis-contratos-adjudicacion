import re
import string
import json


# Función para limpiar el texto
import re
import string

# Función para limpiar el texto
def clean_text(text):

    # . Eliminar cadenas que combinan letras con otros símbolos como "/"
    text = re.sub(r'[a-zA-Z]+\/+[a-zA-Z]+', '', text)

    # 1. Eliminar cadenas numéricas largas (más de 15 caracteres y que solo contengan números y espacios)
    text = re.sub(r'\b\d{16,}\b', '', text)  # Elimina cadenas numéricas con más de 15 caracteres (que solo son números y espacios)

    # 2. Eliminar caracteres extraños no alfanuméricos, excepto letras con tildes, números y guiones
    # Usamos una expresión regular que permita letras con tildes y la ñ
    text = re.sub(r'[^\w\sáéíóúÁÉÍÓÚñÑ-]', ' ', text)  # Permitir letras con tildes y la ñ, y guiones

    # 3. Eliminar guiones sueltos (no conectados a letras o números)
    text = re.sub(r'(?<!\w)-|-(?!\w)', ' ', text)  # Eliminar guiones sueltos, pero mantener guiones con letras/números
    
    # 4. Eliminar saltos de línea innecesarios y caracteres de control
    text = re.sub(r'\n+', ' ', text)  # Reemplazar múltiples saltos de línea por un solo espacio
    text = re.sub(r'\r+', ' ', text)  # Eliminar retornos de carro
    
    # 5. Eliminar espacios adicionales
    text = re.sub(r'\s+', ' ', text)  # Reemplazar múltiples espacios por uno solo
    
    # 6. Eliminar comas y puntos que estén solos o con espacios innecesarios
    text = re.sub(r'(?<=\w)[,.]\s*', r'\g<0>', text)  # Comas y puntos deben estar inmediatamente después de una letra
    
    # 7. Convertir a minúsculas sin afectar las letras con tildes
    text = text.lower()

    
    
    # 8. Eliminar cadenas con más de 17 caracteres
    text = ' '.join([word for word in text.split() if len(word) <= 17])  # Eliminar palabras con más de 17 caracteres
    
    # 9. Eliminar palabras individuales como 'y', 'o', 'a' y otros caracteres innecesarios
    #text = ' '.join([word for word in text.split() if len(word) > 1 or word in ['y', 'o', 'a']])

    # 10. Validar que el texto no esté vacío después de la limpieza
    text = text.strip()
    if len(text) == 0:
        return None  # Si el texto está vacío después de la limpieza, devolvemos None
    
    return text

# Leer el archivo JSONL y procesarlo
input_file = 'actas_extraidas.jsonl'
output_file = 'actas_limpias.jsonl'

processed_data = []

# Leer y procesar el archivo JSONL
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        acta = json.loads(line)
        
        # Suponemos que el texto en bruto está en 'texto'
        if 'texto' in acta:
            cleaned_text = clean_text(acta['texto'])  # Limpiar el texto
            
            # Solo procesamos si el texto limpio no es None (es decir, no está vacío) y tiene una longitud mayor que un umbral mínimo
            if cleaned_text and len(cleaned_text) > 300:  # El texto debe tener más de 300 caracteres
                processed_data.append({
                    'archivo': acta['archivo'],
                    'texto_limpio': cleaned_text
                })

# Guardar el archivo de actas limpias en JSONL
with open(output_file, 'w', encoding='utf-8') as outfile:
    for data in processed_data:
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"Archivo procesado y guardado en: {output_file}")
