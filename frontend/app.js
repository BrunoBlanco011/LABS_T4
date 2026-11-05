const $ = (s) => document.querySelector(s);
const EJEMPLOS = ['EZLN', 'temblor muy fuerte en la costa del Pacífico', 'felino con manchas más grande de América',
  'enfermedades que contagian los mosquitos', 'presas hidroeléctricas en el río Grijalva', 'escritores chiapanecos'];
const COLOR_ETAPA = {bm25: 'var(--bm25)', denso: 'var(--denso)', rrf: '#999', cross_encoder: 'var(--hibrido)'};
const NOMBRE_ETAPA = {bm25: 'BM25', denso: 'embeddings', rrf: 'RRF', cross_encoder: 'cross-encoder'};
const ENTIDADES = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ENTIDADES[c]);
const sinAcentos = (s) => s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();

// Marca en el fragmento las palabras que empiezan como alguna palabra de la consulta (>3 letras).
function resaltar(texto, consulta) {
  const raices = sinAcentos(consulta).split(/[^a-z0-9ñ]+/).filter((t) => t.length > 3).map((t) => t.slice(0, 5));
  return esc(texto).split(/(\s+)/).map((w) => {
    const base = sinAcentos(w).replace(/[^a-z0-9ñ]/g, '');
    return raices.some((r) => base.startsWith(r)) ? `<mark>${w}</mark>` : w;
  }).join('');
}

function columna(r, consulta) {
  const total = Object.values(r.etapas_ms).reduce((a, b) => a + b, 0) || 1;
  const barras = Object.entries(r.etapas_ms).map(([e, v]) =>
    `<span style="width:${(100 * v / total).toFixed(1)}%;background:${COLOR_ETAPA[e]}" title="${NOMBRE_ETAPA[e]}: ${v.toFixed(1)} ms"></span>`).join('');
  const leyenda = Object.entries(r.etapas_ms).map(([e, v]) => `${NOMBRE_ETAPA[e]} ${v.toFixed(1)}`).join(' · ');
  const items = r.resultados.length ? r.resultados.map((d, i) => `
    <li><div class="fila"><button class="titulo" data-id="${d.id}">${i + 1}. ${esc(d.titulo)}</button>
      <span class="score">${d.score.toFixed(3)}</span></div>
      <span class="cat">${esc(d.categoria)} · ${d.id}</span>
      <p class="frag">${resaltar(d.fragmento, consulta)}</p></li>`).join('')
    : '<p class="vacio">Sin resultados: ningún documento comparte términos con la consulta.</p>';
  return `<article class="motor" data-m="${r.motor}"><header><h2>${esc(r.etiqueta)}</h2>
    <div class="lat">latencia del motor <b>${r.latencia_ms.toFixed(1)} ms</b></div>
    <div class="etapas">${barras}</div><div class="leyenda">${leyenda} ms</div></header><ol>${items}</ol></article>`;
}

async function buscar(consulta) {
  if (!consulta) return;
  const modo = $('#modo').value;
  const url = modo === 'comparar'
    ? `/api/comparar?k=5&q=${encodeURIComponent(consulta)}`
    : `/api/buscar?k=10&motor=${modo}&q=${encodeURIComponent(consulta)}`;
  $('#estado').textContent = 'Buscando…';
  const t0 = performance.now();
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const datos = await resp.json();
    const ida = performance.now() - t0;
    const lista = modo === 'comparar' ? datos.motores : [datos];
    $('#resultados').innerHTML = lista.map((r) => columna(r, consulta)).join('');
    $('#estado').textContent = `Ida y vuelta navegador ↔ API: ${ida.toFixed(0)} ms · tiempo en el servidor: ${resp.headers.get('X-Tiempo-Servidor-ms')} ms`;
    history.replaceState(null, '', `?q=${encodeURIComponent(consulta)}&modo=${modo}`);
  } catch (e) {
    $('#estado').textContent = 'Error: ' + e.message;
  }
}

async function abrirDocumento(id) {
  const d = await (await fetch(`/api/documento/${id}`)).json();
  $('#dlg-titulo').textContent = d.titulo;
  $('#dlg-meta').innerHTML = `${d.id} · ${esc(d.categoria)} · ${d.texto.split(/\s+/).length} palabras · ` +
    `<a href="${esc(d.url)}" target="_blank" rel="noopener">fuente en Wikipedia</a>`;
  $('#dlg-texto').innerHTML = `<p>${esc(d.texto)}</p>`;
  $('#dialogo').showModal();
}

$('#formulario').addEventListener('submit', (e) => { e.preventDefault(); buscar($('#consulta').value.trim()); });
$('#modo').addEventListener('change', () => buscar($('#consulta').value.trim()));
$('#resultados').addEventListener('click', (e) => { const b = e.target.closest('.titulo'); if (b) abrirDocumento(b.dataset.id); });
$('#dlg-cerrar').addEventListener('click', () => $('#dialogo').close());
$('#dialogo').addEventListener('click', (e) => { if (e.target.id === 'dialogo') $('#dialogo').close(); });
EJEMPLOS.forEach((q) => {
  const b = document.createElement('button');
  b.type = 'button'; b.textContent = q;
  b.onclick = () => { $('#consulta').value = q; buscar(q); };
  $('#ejemplos').appendChild(b);
});

(async () => {
  const c = await (await fetch('/api/corpus')).json();
  $('#info-corpus').textContent = `${c.documentos} artículos de Wikipedia · ${c.palabras_total.toLocaleString('es-MX')} palabras ` +
    `(${c.palabras_min}–${c.palabras_max} por documento) · motores: BM25, embeddings mpnet, híbrido RRF + cross-encoder`;
  const p = new URLSearchParams(location.search);
  if (p.get('modo')) $('#modo').value = p.get('modo');
  if (p.get('q')) { $('#consulta').value = p.get('q'); buscar(p.get('q')); }
})();
