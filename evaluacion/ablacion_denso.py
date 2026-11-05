"""Ablación: ¿cómo representar un documento largo (≈400 palabras) con un encoder de 128 tokens?

Se comparan 5 variantes con las 19 consultas de qrels.json (nDCG@5 medio). El motor usa
'max pasajes', elegido antes de evaluar; las demás se reportan solo como referencia.
Uso: python evaluacion/ablacion_denso.py  -> resultados/ablacion_denso.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'backend')); sys.path.insert(0, str(RAIZ / 'evaluacion'))
from motores import MotorDenso, cargar_corpus, limpiar  # noqa: E402
from evaluar_calidad import QR, ndcg  # noqa: E402

corpus = cargar_corpus(); den = MotorDenso(corpus)
ids = [d['id'] for d in corpus]; P, pdoc = den.indice, den.pasaje_doc
primero = {i: np.where(pdoc == i)[0][0] for i in ids}
media = den._normalizar_l2(np.stack([P[pdoc == i].mean(0) for i in ids]))
trunc = den._normalizar_l2(den._codificar([limpiar(d['titulo'] + '. ' + d['texto']) for d in corpus]))
rk = lambda s: [d for d, _ in sorted(s.items(), key=lambda x: -x[1])]
filas = []
for c in QR:
    q = den.vectorizar_consulta(c['q']); cos = P @ q
    por_doc = {i: np.sort(cos[pdoc == i]) for i in ids}
    variantes = {
        'Máximo sobre pasajes (motor)': {i: v[-1] for i, v in por_doc.items()},
        'Media de los 2 mejores pasajes': {i: v[-2:].mean() for i, v in por_doc.items()},
        'Embedding medio de los pasajes': dict(zip(ids, media @ q)),
        'Solo el primer pasaje': {i: cos[primero[i]] for i in ids},
        'Documento truncado a 128 tokens': dict(zip(ids, trunc @ q)),
    }
    for n, s in variantes.items():
        filas.append({'variante': n, 'tipo': c['tipo'], 'nDCG@5': ndcg(rk(s), c['rel'])})
df = pd.DataFrame(filas)
t = df.groupby('variante', sort=False)['nDCG@5'].mean().to_frame('nDCG@5')
t = t.join(df.pivot_table(index='variante', columns='tipo', values='nDCG@5'))
t.to_csv(RAIZ / 'resultados' / 'ablacion_denso.csv', encoding='utf-8')
print(t.round(3).to_string())
