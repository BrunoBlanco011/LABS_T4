"""Calidad de los tres motores con las 19 consultas de evaluacion/qrels.json.

Métricas del Lab 7: nDCG@5 (ganancia 2^rel - 1), MRR y Recall@5. Se agrega P@5 y,
como ablación, el híbrido sin cross-encoder (solo RRF) para separar el efecto de cada etapa.

Uso:  python evaluacion/evaluar_calidad.py
Salida: resultados/calidad_por_consulta.csv, resultados/calidad_resumen.csv, figuras/calidad_*.png
"""
import json, math, sys, time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'backend'))
from motores import cargar_corpus, construir_motores  # noqa: E402

QR = json.load(open(RAIZ / 'evaluacion' / 'qrels.json', encoding='utf-8'))['consultas']


def ndcg(rank, rel, k=5):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(rank[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr(rank, rel):
    return next((1 / (i + 1) for i, d in enumerate(rank) if rel.get(d, 0) > 0), 0.0)


def recall(rank, rel, k=5):
    r = {d for d, g in rel.items() if g > 0}
    return len(r & set(rank[:k])) / len(r)


def precision(rank, rel, k=5):
    return sum(rel.get(d, 0) > 0 for d in rank[:k]) / k


def main():
    motores, _ = construir_motores(cargar_corpus())
    sistemas = {
        'BM25': lambda q: motores['bm25'].buscar(q, 10)[0],
        'Embeddings': lambda q: motores['denso'].buscar(q, 10)[0],
        'Híbrido (solo RRF)': lambda q: motores['hibrido'].buscar(q, 10, rerank=False)[0],
        'Híbrido RRF + CE': lambda q: motores['hibrido'].buscar(q, 10)[0],
    }
    filas = []
    for nombre, f in sistemas.items():
        for c in QR:
            t0 = time.perf_counter()
            rank = [d for d, _ in f(c['q'])]
            ms = (time.perf_counter() - t0) * 1000
            filas.append({'sistema': nombre, 'consulta': c['q'], 'tipo': c['tipo'],
                          'nDCG@5': ndcg(rank, c['rel']), 'MRR': mrr(rank, c['rel']),
                          'R@5': recall(rank, c['rel']), 'P@5': precision(rank, c['rel']),
                          'devueltos': len(rank), 'top5': ' '.join(rank[:5]), 'ms_1usuario': ms})
    df = pd.DataFrame(filas)
    (RAIZ / 'resultados').mkdir(exist_ok=True)
    df.to_csv(RAIZ / 'resultados' / 'calidad_por_consulta.csv', index=False, encoding='utf-8')

    orden = list(sistemas)
    res = df.groupby('sistema')[['nDCG@5', 'MRR', 'R@5', 'P@5']].mean().loc[orden]
    por_tipo = df.pivot_table(index='sistema', columns='tipo', values='nDCG@5').loc[orden]
    res = res.join(por_tipo.rename(columns=lambda t: f'nDCG@5 {t}'))
    res.to_csv(RAIZ / 'resultados' / 'calidad_resumen.csv', encoding='utf-8')
    pd.set_option('display.width', 200)
    print(res.round(3).to_string())
    print('\nnDCG@5 por consulta')
    print(df.pivot_table(index=['tipo', 'consulta'], columns='sistema', values='nDCG@5')[orden].round(3).to_string())

    # figura: nDCG@5 por tipo de consulta
    fig, ax = plt.subplots(figsize=(8, 3.6))
    tipos = ['lexica', 'semantica', 'mixta']
    colores = ['#2f6f9f', '#3f8a5b', '#b9a3c4', '#8a4f9e']
    ancho = 0.2
    for j, (s, col) in enumerate(zip(orden, colores)):
        vals = [por_tipo.loc[s, t] for t in tipos]
        xs = [i + (j - 1.5) * ancho for i in range(len(tipos))]
        ax.bar(xs, vals, ancho, label=s, color=col)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f'{v:.2f}', ha='center', fontsize=7)
    ax.set_xticks(range(len(tipos)), ['Léxicas (6)', 'Semánticas (6)', 'Mixtas (7)'])
    ax.set_ylabel('nDCG@5 medio'); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    (RAIZ / 'figuras').mkdir(exist_ok=True)
    plt.savefig(RAIZ / 'figuras' / 'calidad_por_tipo.png', dpi=200)


if __name__ == '__main__':
    main()
