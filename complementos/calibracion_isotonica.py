"""Rama isotónica de calibración para las 560 unidades de la realización principal.

Reutiliza exactamente el estimador base y las tres particiones internas que se
usaron para la rama sigmoid (guardadas dentro de ``calibrated.pkl``), de modo que
sigmoid e isotónica solo difieren en la función de recalibración.
"""
from pathlib import Path
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import cloudpickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
OFICIAL = ROOT / 'resultados_modelado/oficial/20260920T030103_517c8c6b7906_69c6c7'
ANALISIS = ROOT / 'resultados_modelado/analisis_final'
SALIDA = ROOT / 'resultados_modelado/complementarios/calibracion_isotonica'


def cargar_desarrollo():
    """Devuelve las variables originales y la etiqueta indexadas por row_id."""
    raw = pd.read_csv(ROOT / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    y = raw.Churn.map({'No': 0, 'Yes': 1}).astype(int)
    return raw.drop(columns=['customerID', 'Churn']), y


def ece_uniforme(y, probability, bins=10):
    """ECE con diez intervalos uniformes, igual que la rama sigmoid."""
    y, probability = np.asarray(y), np.asarray(probability)
    indices = np.minimum((probability * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = indices == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(total)


def calibrar_unidad(row, X, y, splits):
    """Ajusta la rama isotónica de una unidad externa y la evalúa en su fold reservado.

    Parameters
    ----------
    row : namedtuple
        Fila de ``unidades.csv`` (realización 0).
    X, y : DataFrame, Series
        Datos originales de los 7.043 clientes indexados por ``row_id``.
    splits : list of dict
        Particiones externas oficiales (``splits.json``).

    Returns
    -------
    (dict, DataFrame)
        Métricas de la unidad y probabilidades isotónicas por cliente evaluado.
    """
    folder = OFICIAL / Path(str(row.attempt_path).replace(chr(92), '/'))
    with (folder / 'calibrated.pkl').open('rb') as file:
        sigmoid = cloudpickle.load(file)
    pred = pd.read_csv(folder / 'predictions.csv', float_precision='round_trip')
    saved = pd.read_csv(folder / 'calibrated_predictions.csv', float_precision='round_trip')
    split = splits[int(row.outer_fold)]
    X_fit, y_fit = X.loc[split['fit_ids']], y.loc[split['fit_ids']]
    X_eval = X.loc[pred.row_id]
    # Las particiones internas guardadas deben coincidir con las de la rama sigmoid.
    info = json.loads((folder / 'calibration_checkpoint.json').read_text(encoding='utf-8'))['data']['info']
    for (fit_pos, cal_pos), registro in zip(sigmoid.cv, info['splits']):
        assert X_fit.index[cal_pos].tolist() == registro['calibration_ids']
    # El sigmoid persistido reproduce sus probabilidades guardadas en este entorno.
    np.testing.assert_allclose(sigmoid.predict_proba(X_eval)[:, 1], saved.probability_sigmoid, atol=1e-10)

    isotonic = CalibratedClassifierCV(clone(sigmoid.estimator), method='isotonic',
                                      cv=sigmoid.cv, ensemble=False, n_jobs=1)
    start = pd.Timestamp.now()
    isotonic.fit(X_fit, y_fit)
    seconds = (pd.Timestamp.now() - start).total_seconds()
    probability = isotonic.predict_proba(X_eval)[:, 1]
    labels = (probability >= 0.5).astype(int)
    y_eval = pred.y_true.to_numpy()
    metrics = {'unit_id': row.unit_id, 'experiment_id': row.experiment_id, 'model': row.model,
               'balance': row.balance, 'optimizer': row.optimizer, 'outer_fold': int(row.outer_fold),
               'iso_brier': brier_score_loss(y_eval, probability), 'iso_ece': ece_uniforme(y_eval, probability),
               'iso_accuracy': accuracy_score(y_eval, labels),
               'iso_precision_positive': precision_score(y_eval, labels, zero_division=0),
               'iso_recall_positive': recall_score(y_eval, labels, zero_division=0),
               'iso_f1_positive': f1_score(y_eval, labels, zero_division=0),
               'iso_roc_auc': roc_auc_score(y_eval, probability),
               'iso_distinct_values': int(np.unique(probability).size),
               'iso_seconds': seconds}
    frame = pd.DataFrame({'unit_id': row.unit_id, 'row_id': pred.row_id,
                          'probability_isotonic': probability})
    return metrics, frame


def ejecutar(n_jobs=4, limite=None):
    """Calcula la rama isotónica para todas las unidades de la realización 0."""
    unidades = pd.read_csv(ANALISIS / 'unidades.csv', float_precision='round_trip')
    principal = unidades[unidades.replicate == 0].sort_values(['experiment_id', 'outer_fold'])
    if limite:
        principal = principal.head(limite)
    X, y = cargar_desarrollo()
    splits = json.loads((OFICIAL / 'splits.json').read_text(encoding='utf-8'))
    salida = Parallel(n_jobs=n_jobs, verbose=5)(delayed(calibrar_unidad)(row, X, y, splits)
                                               for row in principal.itertuples())
    metricas = pd.DataFrame([m for m, _ in salida])
    probabilidades = pd.concat([f for _, f in salida], ignore_index=True)
    return metricas, probabilidades


if __name__ == '__main__':
    import sys
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else None
    metricas, probabilidades = ejecutar(limite=limite)
    SALIDA.mkdir(parents=True, exist_ok=True)
    sufijo = '_prueba' if limite else ''
    metricas.to_csv(SALIDA / f'unidades_isotonica{sufijo}.csv', index=False)
    probabilidades.to_csv(SALIDA / f'probabilidades_isotonica{sufijo}.csv.gz', index=False)
    print(metricas.describe().T)
