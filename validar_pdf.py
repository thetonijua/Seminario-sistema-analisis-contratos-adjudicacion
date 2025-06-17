import os
from PyPDF2 import PdfReader

carpeta = "actas_descargadas"  # Cambia al nombre de tu carpeta

def es_pdf_valido(ruta):
    try:
        with open(ruta, "rb") as f:
            reader = PdfReader(f)
            # Verificamos que tenga al menos una página y texto legible
            if len(reader.pages) == 0:
                return False
            texto = reader.pages[0].extract_text()
            return bool(texto and texto.strip())
    except Exception as e:
        return False

# Recorre todos los archivos de la carpeta
for archivo in os.listdir(carpeta):
    ruta_completa = os.path.join(carpeta, archivo)

    if not archivo.lower().endswith(".pdf") or not es_pdf_valido(ruta_completa):
        print(f"❌ Eliminando archivo inválido o no legible: {archivo}")
        os.remove(ruta_completa)
    else:
        print(f"✔ Archivo válido: {archivo}")
