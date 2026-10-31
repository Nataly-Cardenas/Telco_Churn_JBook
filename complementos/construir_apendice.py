"""Construye CODIGO_EXPERIMENTO.ipynb a partir de source_snapshot.json con el que se entrenaron las 112 configuraciones.

El código se copia sin modificar; solo se bajan un nivel los encabezados Markdown para
que el apéndice tenga un único título principal dentro del libro.
"""
from pathlib import Path
import hashlib
import json

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'resultados_modelado/oficial/20260920T030103_517c8c6b7906_69c6c7/source_snapshot.json'
MARKDOWN = {'A00-alcance', 'A14-calibracion', 'A19-optimizacion', 'A25-pendientes'}

celdas = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
huella = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
nb = nbformat.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.cells.append(nbformat.v4.new_markdown_cell(f"""# Apéndice: código del entrenamiento de las 112 configuraciones

Las tablas y figuras del modelado avanzado leen resultados ya calculados. Este apéndice
muestra el código que los produjo: las {len(celdas)} celdas que quedaron guardadas en
`source_snapshot.json` en el momento de ejecutarse (SHA-256 `{huella[:16]}…`). Se reproducen tal
cual; lo único que cambia es el nivel de los encabezados.

Las celdas no se ejecutan al construir el libro. Volver a correrlas desde cero implica unas
40.000 evaluaciones de candidatos y varios días de cómputo en un portátil, por eso el flujo
normal es verificar los artefactos con `python verificar_reproduccion.py` y ejecutar
`MODELADO_AVANZADO.ipynb` sobre ellos. Si se quiere repetir el entrenamiento, el orden es el de este
apéndice: importaciones y configuración, datos y particiones, registro, Pipeline, espacios,
evaluador, optimizadores, validación anidada y, al final, `execute_official_block`.

Los textos Markdown intercalados son los de ese momento; algunos anuncian resultados
"pendientes" que hoy están en el capítulo de modelado.""",
                                             metadata={'id': 'X00-presentacion'}))
nb.cells.append(nbformat.v4.new_code_cell(
    "import hashlib, json\nfrom pathlib import Path\n\n"
    "SNAPSHOT = Path('resultados_modelado/oficial/20260920T030103_517c8c6b7906_69c6c7/source_snapshot.json')\n"
    f"assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == '{huella}'\n"
    "celdas = json.loads(SNAPSHOT.read_text(encoding='utf-8'))\n"
    "print(len(celdas), 'celdas; huella verificada')",
    metadata={'id': 'X01-huella'}))
for celda in celdas:
    fuente = ''.join(celda['source']) if isinstance(celda['source'], list) else celda['source']
    if celda['id'] in MARKDOWN or celda['id'].endswith('-texto'):
        fuente = '\n'.join('#' + l if l.startswith('#') else l for l in fuente.splitlines())
        nb.cells.append(nbformat.v4.new_markdown_cell(fuente, metadata={'id': celda['id']}))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(fuente, metadata={'id': celda['id']}))
for i, c in enumerate(nb.cells):
    c['id'] = c.metadata.pop('id')
nbformat.write(nb, ROOT / 'CODIGO_EXPERIMENTO.ipynb')
print('celdas escritas:', len(nb.cells))
