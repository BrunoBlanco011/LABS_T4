"""Convierte data/fuentes.csv en las filas de la tabla del anexo (reporte/fuentes_tabla.tex)."""
import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ESC = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#'}


def esc(s):
    return ''.join(ESC.get(c, c) for c in s)


filas = list(csv.DictReader(open(RAIZ / 'data' / 'fuentes.csv', encoding='utf-8')))
lineas = [f"{r['id']} & {esc(r['titulo'])} & {r['categoria']} & {r['palabras']} & {r['revision']} & "
          f"\\url{{{r['url'].replace('%', chr(92) + '%')}}} \\\\" for r in filas]   # \% dentro de \url
(RAIZ / 'reporte' / 'fuentes_tabla.tex').write_text('\n'.join(lineas) + '\n', encoding='utf-8')
print(len(filas), 'fuentes; consultadas el', filas[0]['consultado'])
