# Seminario-sistema-analisis-contratos-adjudicacion
 Desarrollo de sistema de analisis y deteccion de riesgos legales en resoluciones de adjudicacion de licitaciones

 ## Cómo configurar el entorno

1. Crear entorno virtual:

    python3 -m venv myenv 

    Version utilizada: 3.10

2. Activar el entorno:

- En Windows:
  ```
  myenv\Scripts\activate
  ```
- En macOS/Linux:
  ```
  source myenv/bin/activate
  ```

3. Instalar dependencias:

    pip install -r requirements.txt


4. Ejecutar el sistema:

    modulos/main.py

    ---

  - Ejecutar análisis aleatorio de una resolución
      
    python modulos/main.py --random --pool_jsonl outputs/pool_candidates.jsonl

  - Ejecutar sobre un archivo específico dentro del JSONL

    python modulos/main.py --match "NOMBRE_DEL_ARCHIVO.pdf" outputs/pool_candidates.jsonl

5. Descripción breve del funcionamiento

- Carga una resolución desde `actas_limpias.jsonl`.
- El Planner decide si recuperar precedentes del Gold Set, ajustar k o evaluar riesgos.
- FAISS recupera precedentes mediante embeddings con `paraphrase-mpnet-base-v2`.
- El Analista genera un JSON con riesgos, evidencias y recomendaciones.
- El Crítico valida evidencia literal, catálogo y formato JSON.
- Se aplican reglas anti–falsos-positivos para coherencia.
- Opcionalmente, los resultados se guardan en `outputs/pool_candidates.jsonl`.



