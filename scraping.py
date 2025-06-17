import os
import time
import random
import requests
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from PyPDF2 import PdfReader

# Configura carpeta de descargas
carpeta_descargas = os.path.abspath("actas_descargadas")
os.makedirs(carpeta_descargas, exist_ok=True)

# Configura Chrome con carpeta de descargas automática
chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": carpeta_descargas,
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True
})
chrome_options.add_argument("--headless=new")  # Puedes quitar esta línea si quieres ver el navegador
driver = webdriver.Chrome(options=chrome_options)

# Constantes
TICKET = "0730E88F-4F7F-4C45-B25C-9C224B1A8AF9"
BASE_API = "https://api.mercadopublico.cl/servicios/v1/publico/licitaciones.json"

def random_fecha():
    inicio = datetime(2014, 1, 1)
    fin = datetime(2025, 5, 14)
    delta = fin - inicio
    dias = random.randint(0, delta.days)
    return (inicio + timedelta(days=dias)).strftime("%d%m%Y")

def obtener_licitaciones(fecha):
   
    try:
        r = requests.get(f"{BASE_API}?fecha={fecha}&ticket={TICKET}", timeout=30)
        r.raise_for_status()
        return r.json().get("Listado", [])
    except:
        return []

def obtener_url_acta(codigo):
    try:
        r = requests.get(f"{BASE_API}?codigo={codigo}&ticket={TICKET}", timeout=30)
        r.raise_for_status()
        listado = r.json().get("Listado", [])
        if listado:
            return listado[0].get("Adjudicacion", {}).get("UrlActa")
    except:
        return None
    return None

def es_pdf_legible(ruta_pdf):
    try:
        with open(ruta_pdf, "rb") as f:
            reader = PdfReader(f)
            if len(reader.pages) == 0:
                return False
            texto = reader.pages[0].extract_text()
            return bool(texto and texto.strip())
    except:
        return False

# Archivos válidos
def contar_descargados_validos():
    return len([f for f in os.listdir(carpeta_descargas) if f.endswith(".pdf")])

descargados = contar_descargados_validos()

while descargados < 3000:
    fecha = random_fecha()
    print(f"📅 Buscando licitaciones en fecha {fecha}")
    licitaciones = obtener_licitaciones(fecha)

    for lic in licitaciones:
        if descargados >= 3000:
            break

        codigo = lic.get("CodigoExterno")
        if not codigo:
            continue

        url_acta = obtener_url_acta(codigo)
        if not url_acta:
            continue

        try:
            driver.get(url_acta)
            time.sleep(3)  # esperar a que cargue la tabla

            filas = driver.find_elements(By.XPATH, "//table[@id='DWNL_grdId']//tr")[1:]  # Ignora encabezado
            for fila in filas:
                tipo = fila.find_element(By.XPATH, "./td[3]/span").text.strip()
                nombre = fila.find_element(By.XPATH, "./td[2]/span").text.strip()
                if "Resolución/Decreto Adjudicación" in tipo:
                    # Verifica si ya se ha descargado
                    if any(nombre in f for f in os.listdir(carpeta_descargas)):
                        print(f"✅ Ya descargado: {nombre} ({tipo})")
                        continue

                    print(f"⬇️ Descargando {nombre} ({tipo})")
                    lupa = fila.find_element(By.XPATH, ".//input[@type='image']")
                    lupa.click()
                    time.sleep(4)  # Esperar a que se descargue el archivo

                    # Verificar que el archivo existe y es legible
                    archivo_path = os.path.join(carpeta_descargas, nombre)
                    if os.path.exists(archivo_path):
                        if not es_pdf_legible(archivo_path):
                            print(f"❌ Archivo no legible, eliminado: {nombre}")
                            os.remove(archivo_path)
                        else:
                            print(f"✔ PDF válido: {nombre}")
                    break

        except Exception as e:
            print(f"⚠️ Error con {codigo}: {e}")
            continue

        descargados = contar_descargados_validos()
        print(f"📁 Total descargados válidos: {descargados}/3000")

driver.quit()
print("✅ Descarga completada.")
