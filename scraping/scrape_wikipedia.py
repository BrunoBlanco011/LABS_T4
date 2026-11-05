"""Construye el corpus por web scraping de Wikipedia en espanol.

Descarga el HTML de cada articulo (no usa la API de texto plano), extrae los
parrafos del cuerpo con BeautifulSoup, quita las llamadas a nota [1], y guarda
los primeros parrafos hasta ~350 palabras (minimo 100).

Uso:  python scraping/scrape_wikipedia.py
Salida: data/corpus.json  y  data/fuentes.csv
"""
import csv, json, re, time, sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests
try:                                   # usa el almacen de certificados del sistema (Windows)
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parents[1]
BASE = 'https://es.wikipedia.org/wiki/'
UA = {'User-Agent': 'UPCh-MineriaDatos-Corpus/1.0 (proyecto academico; requests)'}
MIN_PALABRAS, MAX_PALABRAS = 100, 350
CACHE = RAIZ / 'scraping' / 'cache_html'
CACHE.mkdir(parents=True, exist_ok=True)

# (categoria, titulo del articulo en es.wikipedia.org)
ARTICULOS = [
    # --- lugares y turismo ---
    ('turismo', 'Cañón del Sumidero'), ('turismo', 'Zona arqueológica de Palenque'),
    ('turismo', 'San Cristóbal de las Casas'), ('turismo', 'Parque nacional Lagunas de Montebello'),
    ('turismo', 'Cascadas de Agua Azul'), ('turismo', 'Bonampak'), ('turismo', 'Yaxchilán'),
    ('turismo', 'Toniná'), ('turismo', 'Chiapa de Corzo'), ('turismo', 'Comitán de Domínguez'),
    ('turismo', 'Tuxtla Gutiérrez'), ('turismo', 'Tapachula'), ('naturaleza', 'Alouatta pigra'),
    ('turismo', 'Sima de las Cotorras'), ('turismo', 'Misol-Ha'), ('turismo', 'Izapa'),
    # --- naturaleza ---
    ('naturaleza', 'Selva Lacandona'), ('naturaleza', 'Reserva de la biosfera Montes Azules'),
    ('naturaleza', 'Reserva de la Biosfera El Triunfo'), ('naturaleza', 'Volcán Tacaná'),
    ('naturaleza', 'El Chichón'), ('naturaleza', 'Río Grijalva'), ('naturaleza', 'Río Usumacinta'),
    ('naturaleza', 'Pharomachrus mocinno'), ('naturaleza', 'Panthera onca'),
    ('naturaleza', 'Tapirus bairdii'), ('naturaleza', 'Ara macao'), ('naturaleza', 'Crocodylus acutus'),
    # --- desastres, salud y ambiente ---
    ('riesgos', 'Terremoto de Chiapas de 2017'), ('riesgos', 'Huracán Stan'),
    ('riesgos', 'Dengue'), ('riesgos', 'Paludismo'), ('riesgos', 'Deforestación'),
    ('riesgos', 'Sequía'),
    # --- cultura y sociedad ---
    ('cultura', 'Parachico'), ('cultura', 'Marimba'), ('cultura', 'Pueblo tsotsil'),
    ('cultura', 'Pueblo tseltal'), ('cultura', 'Lacandones'), ('cultura', 'Zoque (etnia)'),
    ('cultura', 'Pox'), ('cultura', 'Ámbar de Chiapas'),
    ('cultura', 'Ejército Zapatista de Liberación Nacional'), ('cultura', 'Rosario Castellanos'),
    ('cultura', 'Jaime Sabines'), ('cultura', 'Tascalate'),
    # --- economia e infraestructura ---
    ('economia', 'Café de Chiapas'), ('economia', 'Theobroma cacao'),
    ('economia', 'Chicoasén'), ('economia', 'Presa La Angostura'),
    ('economia', 'Tren Maya'), ('economia', 'Soconusco'),
    # --- educacion y tecnologia ---
    ('tecnologia', 'Universidad Autónoma de Chiapas'), ('tecnologia', 'Inteligencia artificial'),
    ('tecnologia', 'Aprendizaje automático'), ('tecnologia', 'Minería de datos'),
    ('tecnologia', 'Procesamiento de lenguajes naturales'), ('tecnologia', 'Motor de búsqueda'),
    ('tecnologia', 'Robótica'), ('tecnologia', 'Red neuronal artificial'),
]

RE_NOTA = re.compile(r'\[\s*(\d+|nota \d+|[a-z]|cita requerida|sic)\s*\]', re.I)
RE_ESP = re.compile(r'\s+')


def extraer(html):
    soup = BeautifulSoup(html, 'html.parser')
    titulo = soup.select_one('#firstHeading').get_text(' ', strip=True)
    cuerpo = soup.select_one('#mw-content-text .mw-parser-output')
    for basura in cuerpo.select('table, .reference, .mw-editsection, style, sup, .noprint'):
        basura.decompose()
    rev = re.search(r'"wgRevisionId":(\d+)', html)
    parrafos, n = [], 0
    # el HTML actual (Parsoid) envuelve cada seccion en <section>: se recorren todos los <p> en orden
    for p in cuerpo.find_all('p'):
        if p.find_parent(['table', 'blockquote', 'figure', 'li']):
            continue
        t = RE_ESP.sub(' ', RE_NOTA.sub('', p.get_text(' ', strip=True))).strip()
        t = re.sub(r'\s+([,.;:)])', r'\1', t).replace('( ', '(')
        if len(t.split()) < 8:
            continue
        parrafos.append(t); n += len(t.split())
        if n >= MAX_PALABRAS:
            break
    return titulo, ' '.join(parrafos), int(rev.group(1)) if rev else None


def main():
    sesion = requests.Session(); sesion.headers.update(UA)
    corpus, fuentes, hoy = [], [], date.today().isoformat()
    for i, (cat, art) in enumerate(ARTICULOS, 1):
        url = BASE + quote(art.replace(' ', '_'))
        cache = CACHE / (re.sub(r'[^\w]+', '_', art) + '.html')
        if cache.exists():                     # HTML ya descargado: no se vuelve a pedir
            html, url_final = cache.read_text(encoding='utf-8'), url
        else:
            r = sesion.get(url, timeout=30)
            if r.status_code != 200:
                print(f'[{r.status_code}] {art}', file=sys.stderr); continue
            html, url_final = r.text, r.url
            cache.write_text(html, encoding='utf-8'); time.sleep(1.0)   # cortesia con el servidor
        titulo, texto, rev = extraer(html)
        npal = len(texto.split())
        if npal < MIN_PALABRAS:
            print(f'[corto {npal}] {art}', file=sys.stderr); continue
        doc_id = f'd{len(corpus) + 1:02d}'
        corpus.append({'id': doc_id, 'titulo': titulo, 'categoria': cat, 'texto': texto, 'url': url_final})
        fuentes.append({'id': doc_id, 'titulo': titulo, 'categoria': cat, 'palabras': npal,
                        'url': url_final, 'revision': rev, 'consultado': hoy})
        print(f'{doc_id} {npal:4d} pal  {titulo}')
    (RAIZ / 'data').mkdir(exist_ok=True)
    with open(RAIZ / 'data' / 'corpus.json', 'w', encoding='utf-8') as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=1)
    with open(RAIZ / 'data' / 'fuentes.csv', 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fuentes[0])); w.writeheader(); w.writerows(fuentes)
    print(f'\n{len(corpus)} documentos guardados')


if __name__ == '__main__':
    main()
