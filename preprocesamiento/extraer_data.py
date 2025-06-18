import os
import fitz 
import json

carpeta_pdfs = "actas_descargadas"
archivo_salida = "actas_extraidas.jsonl"

def extraer_texto_pdf(ruta_pdf):
    try:
        doc = fitz.open(ruta_pdf)
        texto = ""
        for pagina in doc:
            texto += pagina.get_text()
        doc.close()
        return limpiar_texto(texto)
    except Exception as e:
        print(f"❌ Error al procesar {ruta_pdf}: {e}")
        return ""

def limpiar_texto(texto):
    texto = texto.replace("\xa0", " ")
    texto = " ".join(texto.split())  # Remueve saltos, espacios duplicados
    return texto.strip()

# Crear archivo JSONL con textos extraídos
with open(archivo_salida, "w", encoding="utf-8") as salida:
    for nombre_pdf in os.listdir(carpeta_pdfs):
        if not nombre_pdf.lower().endswith(".pdf"):
            continue

        ruta = os.path.join(carpeta_pdfs, nombre_pdf)
        texto = extraer_texto_pdf(ruta)

        if texto:
            registro = {
                "archivo": nombre_pdf,
                "texto": texto
            }
            salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            print(f"✔ Extraído: {nombre_pdf}")
        else:
            print(f"⚠ Vacío o ilegible: {nombre_pdf}")

print(f"\n✅ Extracción finalizada. Guardado en: {archivo_salida}")
