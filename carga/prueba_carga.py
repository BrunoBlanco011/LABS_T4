"""Prueba de carga: simula de 1 a 50 usuarios concurrentes contra la API.

Modelo de carga *cerrado*: cada usuario virtual es una corrutina que manda una consulta,
espera la respuesta y manda la siguiente (sin tiempo de espera). La latencia se mide en el
cliente (ida y vuelta HTTP completa) y también se guarda el tiempo que reporta el servidor
(cabecera X-Tiempo-Servidor-ms). La diferencia es el tiempo en cola + red local.

Uso:
    uvicorn app:app --app-dir backend --port 8000      # en otra terminal
    python carga/prueba_carga.py [--url http://127.0.0.1:8000] [--usuarios 1 5 10 25 50]

Salida: resultados/carga_crudo.csv (una fila por petición) y resultados/carga_resumen.csv
"""
import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CONSULTAS = [c['q'] for c in json.load(open(RAIZ / 'evaluacion' / 'qrels.json', encoding='utf-8'))['consultas']]
CONSULTAS += ['café orgánico del Soconusco', 'cascadas de agua turquesa', 'reserva de la biosfera',
              'marimba instrumento nacional', 'huracán que inundó Tapachula', 'redes neuronales profundas',
              'ciudad colonial en las montañas', 'cocodrilo de río', 'mono aullador', 'ámbar con insectos fósiles',
              'universidad pública en Tuxtla', 'motor de búsqueda e índice invertido']


async def usuario(cliente, motor, n, registros, rng, escenario, nusuarios):
    for _ in range(n):
        m = rng.choice(['bm25', 'denso', 'hibrido']) if motor == 'mixto' else motor
        q = rng.choice(CONSULTAS)
        t0 = time.perf_counter()
        try:
            r = await cliente.get('/api/buscar', params={'q': q, 'motor': m, 'k': 10})
            ok, srv = r.status_code == 200, float(r.headers.get('X-Tiempo-Servidor-ms', 'nan'))
        except httpx.HTTPError:
            ok, srv = False, math.nan
        registros.append({'escenario': escenario, 'motor': m, 'usuarios': nusuarios, 'ok': ok,
                          'latencia_ms': (time.perf_counter() - t0) * 1000, 'servidor_ms': srv,
                          't_fin': time.perf_counter()})


async def escenario(url, motor, nusuarios, peticiones_min, semilla):
    registros = []
    limites = httpx.Limits(max_connections=nusuarios + 5, max_keepalive_connections=nusuarios + 5)
    async with httpx.AsyncClient(base_url=url, timeout=120, limits=limites) as cliente:
        for q in CONSULTAS[:5]:                                  # calentamiento (no se mide)
            await cliente.get('/api/buscar', params={'q': q, 'motor': 'hibrido' if motor == 'mixto' else motor})
        por_usuario = max(10, math.ceil(peticiones_min / nusuarios))
        t0 = time.perf_counter()
        await asyncio.gather(*[usuario(cliente, motor, por_usuario, registros, random.Random(semilla + i),
                                       motor, nusuarios) for i in range(nusuarios)])
        dur = time.perf_counter() - t0
    for r in registros:
        r['duracion_s'] = dur
    return registros


def resumir(df):
    g = df.groupby(['escenario', 'usuarios'])
    t = g.agg(peticiones=('ok', 'size'), errores=('ok', lambda s: int((~s).sum())),
              p50_ms=('latencia_ms', lambda s: np.percentile(s, 50)),
              p95_ms=('latencia_ms', lambda s: np.percentile(s, 95)),
              p99_ms=('latencia_ms', lambda s: np.percentile(s, 99)),
              media_ms=('latencia_ms', 'mean'), max_ms=('latencia_ms', 'max'),
              servidor_p50_ms=('servidor_ms', 'median'), duracion_s=('duracion_s', 'first'))
    t['rps'] = t['peticiones'] / t['duracion_s']
    return t.reset_index()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8000')
    ap.add_argument('--usuarios', type=int, nargs='+', default=[1, 5, 10, 25, 50])
    ap.add_argument('--motores', nargs='+', default=['bm25', 'denso', 'hibrido'])
    ap.add_argument('--peticiones', type=int, default=300, help='peticiones mínimas por escenario')
    a = ap.parse_args()
    async with httpx.AsyncClient(base_url=a.url) as c:
        salud = (await c.get('/api/salud')).json()
    print('servidor:', salud)
    todos = []
    planes = [(m, u) for m in a.motores for u in a.usuarios] + [('mixto', max(a.usuarios))]
    for motor, u in planes:
        regs = await escenario(a.url, motor, u, a.peticiones, semilla=1000 * u)
        todos += regs
        lat = np.array([r['latencia_ms'] for r in regs])
        print(f'{motor:8s} {u:3d} usuarios  n={len(regs):5d}  p50={np.percentile(lat, 50):8.1f} ms  '
              f'p95={np.percentile(lat, 95):8.1f} ms  rps={len(regs) / regs[0]["duracion_s"]:6.1f}  '
              f'errores={sum(not r["ok"] for r in regs)}', flush=True)
    df = pd.DataFrame(todos)
    (RAIZ / 'resultados').mkdir(exist_ok=True)
    df.to_csv(RAIZ / 'resultados' / 'carga_crudo.csv', index=False)
    res = resumir(df)
    res.to_csv(RAIZ / 'resultados' / 'carga_resumen.csv', index=False)
    print(res.round(1).to_string(index=False))


if __name__ == '__main__':
    asyncio.run(main())
