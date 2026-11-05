# Integración de motores de búsqueda — Tarea 4, Corte 2

**Bruno Blanco Lucio · 233275** · Minería de Datos · UPCh · Docente: Ramses Alejandro Camas Najera

Sistema fullstack que integra los tres motores de búsqueda construidos en las tareas anteriores, sobre un corpus de **60 artículos de Wikipedia en español** obtenido por **web scraping**:

| Motor | Origen | Cómo funciona |
|---|---|---|
| **BM25** | Lab 3 | Lemas de spaCy, índice invertido, *k*₁ = 1.5, *b* = 0.75 |
| **Embeddings** | Tarea 2 | `paraphrase-multilingual-mpnet-base-v2` (768 d), pasajes de 110 palabras, coseno, máximo por documento |
| **Híbrido** | Tarea 3 | RRF (*k* = 60) de BM25 + embeddings (top-20 de cada uno) y re-ranking del top-10 con `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |

El reporte completo (resultados, latencias p50/p95 con hasta 50 usuarios, criterio de decisión y fuentes) está en
[`BLANCO_LUCIO_BRUNO_233275_C2_T4_INTEGRACION.pdf`](BLANCO_LUCIO_BRUNO_233275_C2_T4_INTEGRACION.pdf).

## Estructura

```
scraping/scrape_wikipedia.py   web scraping (requests + BeautifulSoup) -> data/corpus.json, data/fuentes.csv
backend/motores.py             los 3 motores con la misma interfaz buscar(q, k)
backend/app.py                 API REST (FastAPI) + sirve el frontend
frontend/                      HTML/CSS/JS sin dependencias: buscar con un motor o comparar los 3
evaluacion/qrels.json          19 consultas (léxicas, semánticas, mixtas) con relevancia graduada
evaluacion/evaluar_calidad.py  nDCG@5, MRR, R@5, P@5 por motor
evaluacion/ablacion_denso.py   cómo representar documentos largos con un encoder de 128 tokens
carga/prueba_carga.py          1, 5, 10, 25 y 50 usuarios concurrentes (asyncio + httpx) -> p50, p95, p99, req/s
carga/graficar_carga.py        gráficas de la prueba de carga
carga/captura_frontend.py      captura del frontend con Chrome headless (DevTools Protocol)
resultados/                    CSV con todas las cifras del reporte
figuras/                       gráficas y captura del frontend
reporte/                       fuente LaTeX del reporte
```

## Cómo correrlo

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm

# 1. corpus (opcional: data/corpus.json ya viene incluido)
python scraping/scrape_wikipedia.py

# 2. servidor: API + frontend en http://127.0.0.1:8000
uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000

# 3. evaluación de calidad (no necesita el servidor)
python evaluacion/evaluar_calidad.py

# 4. prueba de carga (con el servidor corriendo) y gráficas
python carga/prueba_carga.py
python carga/graficar_carga.py
```

La primera ejecución descarga de Hugging Face los modelos (≈1.1 GB en total). Con GPU (CUDA) se usa automáticamente; en CPU funciona igual, pero más lento.

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/buscar?q=…&motor=bm25\|denso\|hibrido&k=10` | resultados de un motor, con latencia por etapa |
| GET | `/api/comparar?q=…&k=5` | los tres motores con la misma consulta |
| GET | `/api/documento/{id}` | texto completo y URL de la fuente |
| GET | `/api/corpus` | lista de documentos y estadísticas |
| GET | `/api/salud` | estado, dispositivo y tiempos de carga |

Cada respuesta trae la cabecera `X-Tiempo-Servidor-ms`. La documentación interactiva está en `/docs`.

## Corpus y licencia

Los textos son extractos de artículos de [es.wikipedia.org](https://es.wikipedia.org) bajo licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.es). La lista de fuentes, con URL, número de revisión y fecha de consulta, está en [`data/fuentes.csv`](data/fuentes.csv) y en el anexo del reporte.

Uso de IA: ver [`AI_USAGE.md`](AI_USAGE.md).
