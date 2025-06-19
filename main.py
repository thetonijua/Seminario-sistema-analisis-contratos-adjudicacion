# main.py

"""
Sistema de Análisis de Riesgos Legales en Actas de Adjudicación
---------------------------------------------------------------
Este script ejecuta una consulta iterativa utilizando embeddings semánticos,
FAISS y el modelo GPT-3.5 para detectar posibles riesgos legales en documentos
de adjudicación del sistema chileno de compras públicas.
"""

from modulos import iterative_search_and_evaluation

# ================================
# CONFIGURACIÓN DE LA CONSULTA
# ================================

consulta_inicial = (
    "¿Este documento menciona correctamente las leyes actuales sobre licitaciones? "
    "¿Faltan normas importantes o hay indicios de posibles irregularidades legales? Como falta de información"
)

# ================================
# EJECUCIÓN DEL ANÁLISIS
# ================================

print("🧠 Iniciando análisis legal iterativo con LLM...\n")
respuesta_final = iterative_search_and_evaluation(consulta_inicial)

# ================================
# RESULTADO
# ================================

print("\n✅ Análisis completo.")
print("📄 Respuesta final del modelo:")
print("-----------------------------------")
print(respuesta_final)

