"""API REST de los tres motores + servidor del frontend.

    uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000

Endpoints
    GET /api/salud                         estado y tiempos de carga
    GET /api/motores                       motores disponibles
    GET /api/buscar?q=...&motor=bm25&k=10  un motor
    GET /api/comparar?q=...&k=5            los tres motores con la misma consulta
    GET /api/documento/{id}                texto completo de un documento
    GET /api/corpus                        lista de documentos (sin texto) y estadísticas
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from motores import RAIZ, cargar_corpus, construir_motores

ESTADO: dict = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    corpus = cargar_corpus()
    motores, tiempos = construir_motores(corpus)
    for m in motores.values():                       # calentamiento: la 1.a llamada a la GPU es lenta
        m.buscar('calentamiento del motor', 5)
    ESTADO.update(corpus={d['id']: d for d in corpus}, motores=motores, carga_ms=tiempos)
    yield
    ESTADO.clear()


app = FastAPI(title='Integración de motores de búsqueda', version='1.0', lifespan=ciclo_de_vida)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET'], allow_headers=['*'])


@app.middleware('http')
async def tiempo_servidor(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers['X-Tiempo-Servidor-ms'] = f'{(time.perf_counter() - t0) * 1000:.2f}'
    return resp


def _fragmento(texto: str, n=260) -> str:
    return texto if len(texto) <= n else texto[:n].rsplit(' ', 1)[0] + '…'


def _ejecutar(nombre: str, q: str, k: int) -> dict:
    motor = ESTADO['motores'][nombre]
    t0 = time.perf_counter()
    res, etapas = motor.buscar(q, k)
    total = (time.perf_counter() - t0) * 1000
    docs = ESTADO['corpus']
    return {
        'motor': nombre, 'etiqueta': motor.etiqueta, 'latencia_ms': round(total, 2),
        'etapas_ms': {e: round(v, 2) for e, v in etapas.items()},
        'resultados': [{'id': d, 'titulo': docs[d]['titulo'], 'categoria': docs[d]['categoria'],
                        'score': round(s, 4), 'fragmento': _fragmento(docs[d]['texto']),
                        'url': docs[d]['url']} for d, s in res],
    }


def _validar(q: str) -> str:
    q = q.strip()
    if not q:
        raise HTTPException(422, 'La consulta está vacía')
    return q


# Endpoints síncronos (def): FastAPI los corre en su pool de hilos, así una búsqueda lenta
# no bloquea el event loop y varias peticiones avanzan a la vez.
@app.get('/api/buscar')
def buscar(q: str = Query(..., max_length=300), motor: str = 'hibrido', k: int = Query(10, ge=1, le=60)):
    if motor not in ESTADO['motores']:
        raise HTTPException(404, f'Motor desconocido: {motor}. Opciones: {list(ESTADO["motores"])}')
    return {'consulta': q, **_ejecutar(motor, _validar(q), k)}


@app.get('/api/comparar')
def comparar(q: str = Query(..., max_length=300), k: int = Query(5, ge=1, le=60)):
    q = _validar(q)
    return {'consulta': q, 'motores': [_ejecutar(n, q, k) for n in ESTADO['motores']]}


@app.get('/api/motores')
def motores():
    return [{'nombre': n, 'etiqueta': m.etiqueta} for n, m in ESTADO['motores'].items()]


@app.get('/api/documento/{doc_id}')
def documento(doc_id: str):
    if doc_id not in ESTADO['corpus']:
        raise HTTPException(404, 'Documento no encontrado')
    return ESTADO['corpus'][doc_id]


@app.get('/api/corpus')
def corpus():
    docs = list(ESTADO['corpus'].values())
    palabras = [len(d['texto'].split()) for d in docs]
    return {'documentos': len(docs), 'palabras_total': sum(palabras),
            'palabras_min': min(palabras), 'palabras_max': max(palabras),
            'lista': [{k: d[k] for k in ('id', 'titulo', 'categoria', 'url')} for d in docs]}


@app.get('/api/salud')
def salud():
    m = ESTADO['motores']
    return {'ok': True, 'documentos': len(ESTADO['corpus']), 'dispositivo': m['denso'].dispositivo,
            'pasajes_densos': len(m['denso'].pasaje_txt), 'carga_ms': ESTADO['carga_ms']}


FRONT = RAIZ / 'frontend'
app.mount('/static', StaticFiles(directory=FRONT), name='static')


@app.get('/', include_in_schema=False)
def inicio():
    return FileResponse(FRONT / 'index.html')
