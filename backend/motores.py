"""Los tres motores de búsqueda de las tareas anteriores, con una interfaz común.

    motor.buscar(consulta, k) -> (lista de (doc_id, score), dict de tiempos en ms)

* MotorBM25     — Lab 3: BM25 (k1 = 1.5, b = 0.75) sobre lemas de spaCy (preprocesamiento del Lab 7).
* MotorDenso    — Tarea 2: embeddings `paraphrase-multilingual-mpnet-base-v2` + coseno.
                  Los documentos de este corpus (≈400 palabras) superan los 128 tokens que lee el
                  encoder, así que se parten en pasajes y el score del documento es el del mejor pasaje.
* MotorHibrido  — Tarea 3: RRF (k = 60) de BM25 + denso y re-ranking del top-10 con el
                  cross-encoder `mmarco-mMiniLMv2-L12-H384-v1`.

Los modelos viven en la GPU si hay una. Se protegen con un candado (`threading.Lock`): las
peticiones concurrentes entran a un hilo cada una (FastAPI), pero el acceso a la GPU se serializa.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
CORPUS_JSON = RAIZ / 'data' / 'corpus.json'

# ---------------------------------------------------------------- limpieza (Tarea 2)
RE_URL = re.compile(r'https?://\S+')
RE_HTML = re.compile(r'</?[a-zA-Z][^>]*>')
RE_EMOJI = re.compile('[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]')
RE_ESPACIOS = re.compile(r'\s+')


def limpiar(texto: str) -> str:
    """Quita HTML, URLs y emojis; conserva acentos, mayúsculas y stopwords (para el encoder)."""
    t = unicodedata.normalize('NFC', texto)
    t = RE_HTML.sub(' ', t); t = RE_URL.sub(' ', t); t = RE_EMOJI.sub(' ', t)
    return RE_ESPACIOS.sub(' ', t.replace('#', ' ')).strip()


def quitar_acentos(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def cargar_corpus(ruta: Path = CORPUS_JSON) -> list[dict]:
    with open(ruta, encoding='utf-8') as fh:
        return json.load(fh)


def _cronometro():
    t0 = time.perf_counter()
    return lambda: (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- BM25 (Lab 3)
class MotorBM25:
    nombre = 'bm25'
    etiqueta = 'BM25 (léxico)'

    def __init__(self, corpus, k1=1.5, b=0.75):
        import spacy
        # Solo hacen falta tokenizador + lematizador: parser y NER se apagan para ganar velocidad.
        self.nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
        conservar = {'no', 'nunca', 'sin', 'ni', 'muy', 'poco', 'nada', 'tampoco'}
        self.stop = set(self.nlp.Defaults.stop_words) - conservar
        self._candado_nlp = threading.Lock()
        self.k1, self.b = k1, b
        self.ids = [d['id'] for d in corpus]
        textos = [d['titulo'] + '. ' + d['texto'] for d in corpus]
        with self._candado_nlp:
            docs = list(self.nlp.pipe([self._normalizar(t) for t in textos], batch_size=16))
        self.tokens = {i: self._lemas(doc) for i, doc in zip(self.ids, docs)}
        n = len(self.ids)
        self.df = Counter(t for i in self.ids for t in set(self.tokens[i]))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in self.df.items()}
        self.tf = {i: Counter(self.tokens[i]) for i in self.ids}
        self.largo = {i: len(self.tokens[i]) for i in self.ids}
        self.avgdl = sum(self.largo.values()) / n
        # índice invertido: término -> [(doc, frecuencia)], para no recorrer todo el corpus
        self.invertido: dict[str, list[tuple[str, int]]] = {}
        for i in self.ids:
            for t, f in self.tf[i].items():
                self.invertido.setdefault(t, []).append((i, f))

    @staticmethod
    def _normalizar(t: str) -> str:
        t = unicodedata.normalize('NFC', t.lower())
        return RE_ESPACIOS.sub(' ', RE_HTML.sub(' ', RE_URL.sub(' ', t))).strip()

    def _lemas(self, doc) -> list[str]:
        out = []
        for tok in doc:
            if tok.is_punct or tok.is_space or tok.like_num or tok.is_stop:
                continue
            lema = quitar_acentos(tok.lemma_.lower())
            if lema in self.stop or len(lema) <= 2:
                continue
            out.append(lema)
        return out

    def preprocesar(self, texto: str) -> list[str]:
        with self._candado_nlp:
            return self._lemas(self.nlp(self._normalizar(texto)))

    def scores(self, consulta: str) -> dict[str, float]:
        """Score BM25 de cada documento que comparte al menos un término con la consulta."""
        acc: dict[str, float] = {}
        for t in self.preprocesar(consulta):
            for i, f in self.invertido.get(t, ()):
                norm = self.k1 * (1 - self.b + self.b * self.largo[i] / self.avgdl)
                acc[i] = acc.get(i, 0.0) + self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return acc

    def buscar(self, consulta: str, k: int = 10):
        fin = _cronometro()
        s = self.scores(consulta)
        res = sorted(s.items(), key=lambda x: (-x[1], x[0]))[:k]
        return res, {'bm25': fin()}


# ---------------------------------------------------------------- embeddings (Tarea 2)
def pasajes(texto: str, palabras=110, paso=80) -> list[str]:
    """Ventanas de `palabras` con traslape: ~110 palabras caben en los 128 tokens de mpnet."""
    w = texto.split()
    if len(w) <= palabras:
        return [texto]
    return [' '.join(w[i:i + palabras]) for i in range(0, max(1, len(w) - palabras + paso), paso)]


class MotorDenso:
    nombre = 'denso'
    etiqueta = 'Embeddings mpnet (semántico)'
    MODELO = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    def __init__(self, corpus, dispositivo=None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.dispositivo = dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = SentenceTransformer(self.MODELO, device=self.dispositivo)
        self._candado = threading.Lock()
        self.ids = [d['id'] for d in corpus]
        self.pasaje_doc, self.pasaje_txt = [], []
        for d in corpus:
            for p in pasajes(limpiar(d['texto'])):
                self.pasaje_doc.append(d['id'])
                self.pasaje_txt.append(limpiar(d['titulo']) + '. ' + p)   # el título acompaña a cada pasaje
        self.pasaje_doc = np.array(self.pasaje_doc)
        self.indice = self._normalizar_l2(self._codificar(self.pasaje_txt))   # (n_pasajes, 768)

    def _codificar(self, textos):
        with self._candado:
            return self.encoder.encode(textos, convert_to_numpy=True, normalize_embeddings=False,
                                       batch_size=32, show_progress_bar=False)

    @staticmethod
    def _normalizar_l2(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    def vectorizar_consulta(self, texto: str):
        return self._normalizar_l2(self._codificar([limpiar(texto)]))[0]

    def scores_pasajes(self, consulta: str):
        """Coseno de la consulta contra cada pasaje -> (score por documento, mejor pasaje por documento)."""
        q = self.vectorizar_consulta(consulta)
        cos = self.indice @ q                     # vectores unitarios: producto punto = coseno
        mejor_s: dict[str, float] = {}
        mejor_p: dict[str, int] = {}
        for j, (d, s) in enumerate(zip(self.pasaje_doc, cos)):
            if s > mejor_s.get(d, -2.0):
                mejor_s[d], mejor_p[d] = float(s), j
        return mejor_s, mejor_p

    def buscar(self, consulta: str, k: int = 10):
        fin = _cronometro()
        s, _ = self.scores_pasajes(consulta)
        res = sorted(s.items(), key=lambda x: (-x[1], x[0]))[:k]
        return res, {'denso': fin()}


# ---------------------------------------------------------------- híbrido (Tarea 3)
def rrf(rankings, k=60):
    acc: Counter = Counter()
    for r in rankings:
        for pos, d in enumerate(r, start=1):
            acc[d] += 1 / (k + pos)
    return acc


class MotorHibrido:
    nombre = 'hibrido'
    etiqueta = 'Híbrido RRF + cross-encoder'
    MODELO_CE = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

    def __init__(self, corpus, bm25: MotorBM25, denso: MotorDenso, profundidad=20, k_rrf=60, n_rerank=10):
        from sentence_transformers import CrossEncoder
        self.bm25, self.denso = bm25, denso
        self.profundidad, self.k_rrf, self.n_rerank = profundidad, k_rrf, n_rerank
        self.ce = CrossEncoder(self.MODELO_CE, device=denso.dispositivo, max_length=512)
        # El cross-encoder lee título + documento completo (se trunca a 512 tokens). Con solo el
        # mejor pasaje de 110 palabras (sin el inicio del artículo) reordenaba peor.
        self.texto_ce = {d['id']: limpiar(d['titulo'] + '. ' + d['texto']) for d in corpus}
        self._candado = threading.Lock()

    def buscar(self, consulta: str, k: int = 10, rerank: bool = True):
        tiempos = {}
        fin = _cronometro()
        sb = self.bm25.scores(consulta)
        rb = [d for d, _ in sorted(sb.items(), key=lambda x: (-x[1], x[0]))[:self.profundidad]]
        tiempos['bm25'] = fin()

        fin = _cronometro()
        sd, _ = self.denso.scores_pasajes(consulta)
        rd = [d for d, _ in sorted(sd.items(), key=lambda x: (-x[1], x[0]))[:self.profundidad]]
        tiempos['denso'] = fin()

        fin = _cronometro()
        fusion = rrf([rb, rd], k=self.k_rrf)
        orden = sorted(fusion.items(), key=lambda x: (-x[1], x[0]))
        tiempos['rrf'] = fin()
        if not rerank:
            return orden[:k], tiempos

        # Etapa 2: el cross-encoder lee consulta + documento de los n primeros candidatos.
        fin = _cronometro()
        cand = [d for d, _ in orden[:self.n_rerank]]
        pares = [(consulta, self.texto_ce[d]) for d in cand]
        with self._candado:
            ce = self.ce.predict(pares, batch_size=16, show_progress_bar=False)
        reordenados = sorted(zip(cand, map(float, ce)), key=lambda x: (-x[1], x[0]))
        tiempos['cross_encoder'] = fin()
        # los que no entraron al re-ranking conservan el orden RRF, debajo de los re-rankeados
        cola = [(d, s) for d, s in orden[self.n_rerank:]]
        return (reordenados + cola)[:k], tiempos


def construir_motores(corpus=None):
    corpus = corpus or cargar_corpus()
    t = {}
    fin = _cronometro(); bm25 = MotorBM25(corpus); t['bm25'] = fin()
    fin = _cronometro(); denso = MotorDenso(corpus); t['denso'] = fin()
    fin = _cronometro(); hib = MotorHibrido(corpus, bm25, denso); t['hibrido'] = fin()
    return {'bm25': bm25, 'denso': denso, 'hibrido': hib}, t
