# AI_USAGE.md — Tarea 4, Corte 2 (integración de motores de búsqueda)

**Alumno:** Bruno Blanco Lucio · **Matrícula:** 233275
**Materia:** Minería de Datos · **Docente:** Ramses Alejandro Camas Najera

## Herramienta utilizada

| Herramienta | Modelo | Modo de uso |
|---|---|---|
| Claude Code (Anthropic) | Claude Opus 5.5 | Asistente en terminal: escribió el scraper, el backend, el frontend y los scripts de evaluación y carga; los ejecutó en local (RTX 4060) y redactó el reporte a partir de sus salidas. |

Todas las cifras del reporte salen de los CSV de `resultados/`. Revisé el código y las decisiones y me hago responsable de ellas para la defensa oral.

## Decisiones y correcciones

| Parte | Qué dio la IA / qué pasó | Qué se decidió |
|---|---|---|
| Scraping | El primer intento falló por certificados SSL en Windows. | Se usa `truststore` (certificados del sistema). Se guarda el HTML en caché para no volver a pedir páginas, con 1 s de pausa entre peticiones. |
| Scraping | El HTML actual de Wikipedia envuelve las secciones en `<section>` y el parser solo buscaba `<p>` hijos directos: 0 palabras. | Se recorren todos los `<p>` del cuerpo, excepto los que están dentro de tablas, listas o figuras. |
| Corpus | *Palenque* bajó el artículo sobre la palabra (no la zona arqueológica) y *Zoques* era una desambiguación. | Se cambiaron por *Zona arqueológica de Palenque* y *Zoque (etnia)*. |
| Motor denso | El encoder lee 128 tokens y los documentos tienen ≈400 palabras. | Pasajes de 110 palabras con traslape; el score del documento es el del mejor pasaje. La ablación (`evaluacion/ablacion_denso.py`) muestra variantes mejores, pero no se cambió el motor para no ajustarlo con las mismas consultas de evaluación. |
| Cross-encoder | Con el mejor pasaje como entrada, reordenaba mal (*Selva Lacandona* primero para *temblor*). | Lee título + documento completo (truncado a 512 tokens). |
| Concurrencia | Los modelos en GPU y spaCy no son seguros para muchos hilos a la vez. | Un candado por modelo: las peticiones avanzan en paralelo y el acceso a la GPU se serializa. Es la causa principal del crecimiento de la latencia con 50 usuarios, y se reporta así. |
