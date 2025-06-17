# main.py
"""
Sistema de detección de riesgos legales en actas de adjudicación usando RAG y LLM
Desarrollado como prototipo funcional para el informe final del primer semestre.
"""

from modules.iterative_development import iterative_search_and_evaluation

# Consulta inicial a evaluar
consulta = "¿Este documento si menciona las leyes actuales sobre licitaciones, tiene leyes desactualizadas o falta información que podría ser indicio de riesgo?"

print("🧠 Iniciando análisis iterativo sobre actas con LLM...")
respuesta = iterative_search_and_evaluation(consulta)

print("\n✅ Análisis completo.")
print("📄 Respuesta final:")
print(respuesta)
