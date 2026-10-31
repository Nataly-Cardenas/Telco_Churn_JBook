"""Verifica que el entorno y los resultados guardados permiten reproducir el libro.

Uso (desde la carpeta del proyecto, con el entorno virtual activo)::

    python verificar_reproduccion.py           # verificación completa (unos minutos)
    python verificar_reproduccion.py --rapida  # omite las huellas de las 1.680 unidades

Comprueba cuatro cosas:

1. Versiones instaladas frente a ``requirements.txt``.
2. Huellas SHA-256 de los datos de entrada frente al manifiesto de la corrida oficial.
3. Huellas de todos los artefactos registrados en cada ``complete.json`` (modelos,
   predicciones, checkpoints) de las 1.680 unidades oficiales.
4. Que las tablas agregadas que lee el notebook se recalculan desde las unidades.

Git puede convertir finales de línea al clonar en Windows (``core.autocrlf``); por
eso, si una huella de texto no coincide byte a byte, se compara también la
versión con finales LF y CRLF y se informa por separado.
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OFICIAL = ROOT / 'resultados_modelado/oficial/20260920T030103_517c8c6b7906_69c6c7'
ANALISIS = ROOT / 'resultados_modelado/analisis_final'
COMPLEMENTARIOS = ROOT / 'resultados_modelado/complementarios'


def sha256(datos):
    return hashlib.sha256(datos).hexdigest()


def coincide(path, esperado):
    """'exacta', 'fin de línea' o 'NO': compara bytes y variantes LF/CRLF."""
    datos = Path(path).read_bytes()
    if sha256(datos) == esperado:
        return 'exacta'
    lf = datos.replace(b'\r\n', b'\n')
    if esperado in (sha256(lf), sha256(lf.replace(b'\n', b'\r\n'))):
        return 'fin de línea'
    return 'NO'


def revisar_versiones():
    faltan, distintas = [], []
    for linea in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        if '==' not in linea or linea.startswith('#'):
            continue
        nombre, requerida = linea.strip().split('==')
        try:
            instalada = version(nombre)
        except PackageNotFoundError:
            faltan.append(nombre)
            continue
        if instalada != requerida:
            distintas.append(f'{nombre}: requerida {requerida}, instalada {instalada}')
    print(f'[1] Python {sys.version.split()[0]} (la corrida oficial usó 3.12.14)')
    print(f'    Paquetes faltantes: {faltan or "ninguno"}')
    print(f'    Versiones distintas: {distintas or "ninguna"}')
    return not faltan and not distintas


def revisar_entradas():
    manifiesto = json.loads((OFICIAL / 'manifest.json').read_text(encoding='utf-8'))
    estados = {nombre: coincide(ROOT / nombre, esperado)
               for nombre, esperado in manifiesto['protocol']['input_hashes'].items()}
    print(f'[2] Datos de entrada frente al manifiesto: {estados}')
    return all(e != 'NO' for e in estados.values())


def revisar_unidades():
    unidades = pd.read_csv(ANALISIS / 'unidades.csv')
    conteo = {'exacta': 0, 'fin de línea': 0, 'NO': 0, 'faltante': 0}
    for unit_id in unidades.unit_id.unique():
        carpeta = OFICIAL / unit_id
        registro = json.loads((carpeta / 'complete.json').read_text(encoding='utf-8'))
        for relativo, esperado in registro['files'].items():
            archivo = carpeta / relativo.replace(chr(92), '/')
            conteo['faltante' if not archivo.exists() else coincide(archivo, esperado)] += 1
    print(f'[3] Artefactos de {unidades.unit_id.nunique()} unidades: {conteo}')
    return conteo['NO'] == 0 and conteo['faltante'] == 0


def revisar_agregados():
    unidades = pd.read_csv(ANALISIS / 'unidades.csv', float_precision='round_trip')
    principal = unidades[unidades.replicate == 0]
    metricas = ['accuracy', 'precision_positive', 'recall_positive', 'f1_positive', 'roc_auc', 'brier', 'ece']
    resumen = principal.groupby('experiment_id')[metricas].agg(['mean', 'std'])
    resumen.columns = ['_'.join(c) for c in resumen.columns]
    guardado = pd.read_csv(ANALISIS / 'resumen_112.csv', float_precision='round_trip').set_index('experiment_id')
    np.testing.assert_allclose(resumen.sort_index()[guardado.select_dtypes('number').columns],
                               guardado.sort_index().select_dtypes('number'), rtol=1e-12, atol=1e-14)
    esperados = ['calibracion_isotonica/unidades_isotonica.csv', 'multifidelidad/multifidelidad_por_fold.csv',
                 'optimizacion_computacional/comparaciones_por_fold.csv']
    faltan = [e for e in esperados if not (COMPLEMENTARIOS / e).exists()]
    print(f'[4] resumen_112.csv se recalcula desde unidades.csv: sí; complementarios faltantes: {faltan or "ninguno"}')
    return not faltan


if __name__ == '__main__':
    rapida = '--rapida' in sys.argv
    resultados = [revisar_versiones(), revisar_entradas(),
                  True if rapida else revisar_unidades(), revisar_agregados()]
    print('\nResultado:', 'todo verificado' if all(resultados) else 'hay diferencias (ver arriba)')
    sys.exit(0 if all(resultados) else 1)
