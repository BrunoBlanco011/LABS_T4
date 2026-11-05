"""Gráficas de la prueba de carga a partir de resultados/carga_crudo.csv.
Uso: python carga/graficar_carga.py  -> figuras/carga_*.png y resultados/carga_tabla.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
df = pd.read_csv(RAIZ / 'resultados' / 'carga_crudo.csv')
COL = {'bm25': '#2f6f9f', 'denso': '#3f8a5b', 'hibrido': '#8a4f9e'}
NOM = {'bm25': 'BM25', 'denso': 'Embeddings', 'hibrido': 'Híbrido RRF + CE'}
puros = df[df['escenario'] != 'mixto']


def pct(s, p):
    return np.percentile(s, p)


tabla = (puros.groupby(['motor', 'usuarios'])
         .agg(n=('latencia_ms', 'size'), p50=('latencia_ms', lambda s: pct(s, 50)),
              p95=('latencia_ms', lambda s: pct(s, 95)), p99=('latencia_ms', lambda s: pct(s, 99)),
              servidor_p50=('servidor_ms', 'median'), dur=('duracion_s', 'first'),
              errores=('ok', lambda s: int((~s).sum())))
         .reset_index())
tabla['rps'] = tabla['n'] / tabla['dur']
tabla.drop(columns='dur').to_csv(RAIZ / 'resultados' / 'carga_tabla.csv', index=False)

# 1) p50 y p95 contra usuarios concurrentes
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, p in zip(axs, ['p50', 'p95']):
    for m in ['bm25', 'denso', 'hibrido']:
        t = tabla[tabla['motor'] == m]
        ax.plot(t['usuarios'], t[p], 'o-', color=COL[m], label=NOM[m])
        for x, y in zip(t['usuarios'], t[p]):
            ax.annotate(f'{y:.0f}', (x, y), fontsize=7, xytext=(3, 4), textcoords='offset points', color=COL[m])
    ax.set_yscale('log'); ax.set_xlabel('usuarios concurrentes'); ax.set_title(f'Latencia {p} (ms, escala log)')
    ax.set_xticks([1, 5, 10, 25, 50]); ax.grid(alpha=.3, which='both'); ax.spines[['top', 'right']].set_visible(False)
axs[0].set_ylabel('ms'); axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(RAIZ / 'figuras' / 'carga_latencia.png', dpi=200)

# 2) rendimiento (peticiones por segundo)
fig, ax = plt.subplots(figsize=(6, 3.4))
for m in ['bm25', 'denso', 'hibrido']:
    t = tabla[tabla['motor'] == m]
    ax.plot(t['usuarios'], t['rps'], 'o-', color=COL[m], label=NOM[m])
ax.set_yscale('log'); ax.set_xlabel('usuarios concurrentes'); ax.set_ylabel('peticiones / s (log)')
ax.set_xticks([1, 5, 10, 25, 50]); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(RAIZ / 'figuras' / 'carga_rps.png', dpi=200)

# 3) distribución a 50 usuarios (incluye el escenario mixto, separado por motor)
fig, ax = plt.subplots(figsize=(8, 3.4))
grupos, etiquetas, colores = [], [], []
for m in ['bm25', 'denso', 'hibrido']:
    grupos.append(puros[(puros['motor'] == m) & (puros['usuarios'] == 50)]['latencia_ms'])
    etiquetas.append(f'{NOM[m]}\n(solo)'); colores.append(COL[m])
mix = df[df['escenario'] == 'mixto']
for m in ['bm25', 'denso', 'hibrido']:
    grupos.append(mix[mix['motor'] == m]['latencia_ms'])
    etiquetas.append(f'{NOM[m]}\n(mixto)'); colores.append(COL[m])
bp = ax.boxplot(grupos, tick_labels=etiquetas, whis=(5, 95), showfliers=False, patch_artist=True)
for caja, c in zip(bp['boxes'], colores):
    caja.set_facecolor(c); caja.set_alpha(.55)
ax.set_yscale('log'); ax.set_ylabel('ms (log)'); ax.tick_params(axis='x', labelsize=7)
ax.set_title('50 usuarios: bigotes en p5 y p95', fontsize=10); ax.grid(alpha=.3, axis='y', which='both')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(RAIZ / 'figuras' / 'carga_50usuarios.png', dpi=200)

print(tabla.round(1).to_string(index=False))
if len(mix):
    print('\nmixto (50 usuarios) por motor:')
    print(mix.groupby('motor')['latencia_ms'].describe(percentiles=[.5, .95]).round(1).to_string())
