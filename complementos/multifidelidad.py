"""Successive Halving dentro de la validación cruzada anidada oficial.

Para Random Forest, XGBoost y SVM con SMOTE se compara, fold externo por fold
externo, una búsqueda aleatoria a fidelidad completa (24 candidatos) con
Successive Halving (27 candidatos, factor 3). Ambas usan el mismo Pipeline,
las mismas semillas del modelo y del remuestreo que la unidad oficial, los
mismos tres folds internos y el mismo objetivo (F1 de la clase abandono).
"""
from pathlib import Path
import hashlib
import json
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import cloudpickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform, randint, spearmanr, uniform
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OFICIAL = ROOT / 'resultados_modelado/oficial/20260920T030103_517c8c6b7906_69c6c7'
ANALISIS = ROOT / 'resultados_modelado/analisis_final'
SALIDA = ROOT / 'resultados_modelado/complementarios/multifidelidad'
SEMILLA_GLOBAL = 42
FACTOR = 3
CANDIDATOS_SH = 27      # 27 → 9 → 3 → 1 con factor 3
CANDIDATOS_COMPLETOS = 24  # mismo presupuesto de candidatos que la corrida oficial

# Fidelidad: número de árboles en los ensambles y tamaño de muestra en SVM.
ESPACIOS = {
    'random_forest': {
        'recurso': 'classifier__estimator__n_estimators', 'minimo': 12, 'maximo': 324,
        'espacio': {'classifier__estimator__max_depth': [4, 6, 8, 10, 12, 16, None],
                    'classifier__estimator__min_samples_leaf': randint(1, 21),
                    'classifier__estimator__max_features': uniform(0.2, 0.6),
                    'classifier__estimator__criterion': ['gini', 'entropy']}},
    'xgboost': {
        'recurso': 'classifier__estimator__n_estimators', 'minimo': 12, 'maximo': 324,
        'espacio': {'classifier__estimator__max_depth': randint(2, 7),
                    'classifier__estimator__learning_rate': loguniform(0.01, 0.3),
                    'classifier__estimator__subsample': uniform(0.6, 0.4),
                    'classifier__estimator__colsample_bytree': uniform(0.6, 0.4),
                    'classifier__estimator__min_child_weight': loguniform(1, 10)}},
    'svm': {
        'recurso': 'n_samples', 'minimo': 111, 'maximo': None,
        'espacio': {'classifier__estimator__C': loguniform(1e-2, 1e3),
                    'classifier__estimator__gamma': loguniform(1e-3, 1)}},
}


def digest(value):
    """SHA-256 de la serialización JSON canónica usada en el registro oficial."""
    texto = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(texto.encode('utf-8')).hexdigest()


def datos_desarrollo():
    """Variables originales y etiqueta de los 5.634 clientes de desarrollo, en el orden oficial."""
    raw = pd.read_csv(ROOT / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    orden = pd.read_csv(ROOT / 'train_index.csv').iloc[:, 0].to_numpy()
    y = raw.Churn.map({'No': 0, 'Yes': 1}).astype(int).loc[orden]
    return raw.drop(columns=['customerID', 'Churn']).loc[orden], y


def plan_interno(split, X, y, unidad):
    """Reconstruye los tres folds internos oficiales y verifica su huella contra el registro."""
    fit_pos = np.asarray(split['fit_pos'])
    assert X.index[fit_pos].tolist() == split['fit_ids']
    inner = StratifiedKFold(3, shuffle=True, random_state=split['inner_seed'])
    plan = list(inner.split(X.iloc[fit_pos], y.iloc[fit_pos]))
    evaluaciones = next((unidad / 'search/evaluations').iterdir())
    for k, (a, _) in enumerate(plan):
        registro = json.loads((evaluaciones / f'inner_{k}.json').read_text(encoding='utf-8'))
        assert digest(X.index[fit_pos[a]].tolist()) == registro['data']['audit']['fit_ids_hash']
    return plan


def pipeline_oficial(model, fold):
    """Pipeline (sin ajustar) de la unidad oficial random de la realización 0 en ese fold."""
    unidades = pd.read_csv(ANALISIS / 'unidades.csv')
    fila = unidades[(unidades.experiment_id == f'{model}__smote__random') &
                    (unidades.replicate == 0) & (unidades.outer_fold == fold)].iloc[0]
    carpeta = OFICIAL / Path(str(fila.attempt_path).replace(chr(92), '/'))
    with (carpeta / 'pipeline.pkl').open('rb') as file:
        return clone(cloudpickle.load(file)), carpeta


def costo_relativo(cv_results, recurso, maximo):
    """Ajustes equivalentes a fidelidad completa: suma de recurso/recurso_máximo por fit."""
    n_resources = np.asarray(cv_results.get('n_resources', [maximo] * len(cv_results['params'])))
    return float((n_resources / maximo).sum())


def evaluar_fold(model, fold):
    """Ejecuta SH y Random completo en un fold externo y evalúa ambos en el fold reservado."""
    X, y = datos_desarrollo()
    split = json.loads((OFICIAL / 'splits.json').read_text(encoding='utf-8'))[fold]
    pipe, carpeta = pipeline_oficial(model, fold)
    plan = plan_interno(split, X, y, carpeta)
    X_fit, y_fit = X.iloc[split['fit_pos']], y.iloc[split['fit_pos']]
    X_eval, y_eval = X.iloc[split['eval_pos']], y.iloc[split['eval_pos']]
    spec = ESPACIOS[model]
    maximo = spec['maximo'] or min(len(a) for a, _ in plan)
    semilla = int(np.random.SeedSequence([SEMILLA_GLOBAL, 7, fold, list(ESPACIOS).index(model)]).generate_state(1)[0])
    filas = []
    for metodo in ['successive_halving', 'random_completo']:
        if metodo == 'successive_halving':
            busqueda = HalvingRandomSearchCV(pipe, spec['espacio'], n_candidates=CANDIDATOS_SH, factor=FACTOR,
                                             resource=spec['recurso'], min_resources=spec['minimo'],
                                             max_resources=maximo, aggressive_elimination=False, cv=plan,
                                             scoring='f1', refit=True, random_state=semilla, n_jobs=1,
                                             error_score='raise')
        else:
            base = clone(pipe)
            if spec['recurso'] != 'n_samples':
                base.set_params(**{spec['recurso']: maximo})
            busqueda = RandomizedSearchCV(base, spec['espacio'], n_iter=CANDIDATOS_COMPLETOS, cv=plan,
                                          scoring='f1', refit=True, random_state=semilla, n_jobs=1,
                                          error_score='raise')
        inicio = time.perf_counter()
        busqueda.fit(X_fit, y_fit)
        segundos = time.perf_counter() - inicio
        score = busqueda.best_estimator_.decision_function(X_eval) if model == 'svm' else \
            busqueda.best_estimator_.predict_proba(X_eval)[:, 1]
        etiquetas = busqueda.best_estimator_.predict(X_eval)
        resultados = busqueda.cv_results_
        rangos = None
        if metodo == 'successive_halving':
            # Concordancia entre rondas: candidatos evaluados en la ronda k y en la k+1.
            claves = [json.dumps({k: v for k, v in p.items() if k != spec['recurso']}, sort_keys=True, default=str)
                      for p in resultados['params']]
            tabla = pd.DataFrame({'iter': resultados['iter'], 'params': claves,
                                  'score': resultados['mean_test_score']})
            pares = []
            for k in range(tabla.iter.max()):
                a = tabla[tabla.iter == k].set_index('params').score
                b = tabla[tabla.iter == k + 1].set_index('params').score
                comun = a.index.intersection(b.index)
                if len(comun) >= 3:
                    pares.append(spearmanr(a[comun], b[comun]).statistic)
            rangos = float(np.nanmean(pares)) if pares else np.nan
        ajustes = len(resultados['params']) * len(plan)
        filas.append({'model': model, 'outer_fold': fold, 'method': metodo,
                      'search_seconds': segundos, 'candidates': len(resultados['params']),
                      'inner_fits': ajustes,
                      'full_fidelity_fit_equivalents': costo_relativo(resultados, spec['recurso'], maximo) * len(plan),
                      'best_inner_f1': float(busqueda.best_score_),
                      'outer_f1': f1_score(y_eval, etiquetas), 'outer_auc': roc_auc_score(y_eval, score),
                      'rounds': int(max(resultados.get('iter', [0])) + 1),
                      'resource_name': spec['recurso'], 'max_resource': maximo,
                      'final_resource': int(busqueda.best_params_.get(spec['recurso'], maximo))
                      if spec['recurso'] != 'n_samples' else int(maximo),
                      'rung_spearman': rangos,
                      'best_params': json.dumps(busqueda.best_params_, default=str, sort_keys=True),
                      'seed': semilla})
    return filas


def ejecutar(n_jobs=4):
    """Recorre las tres familias y los cinco folds externos de la realización 0."""
    tareas = [(m, f) for m in ESPACIOS for f in range(5)]
    salida = Parallel(n_jobs=n_jobs, verbose=10)(delayed(evaluar_fold)(m, f) for m, f in tareas)
    return pd.DataFrame([fila for filas in salida for fila in filas])


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        print(pd.DataFrame(evaluar_fold(sys.argv[1], int(sys.argv[2]))).T)
    else:
        resultado = ejecutar()
        SALIDA.mkdir(parents=True, exist_ok=True)
        resultado.to_csv(SALIDA / 'multifidelidad_por_fold.csv', index=False)
        print(resultado.groupby(['model', 'method'])[['search_seconds', 'full_fidelity_fit_equivalents',
                                                      'best_inner_f1', 'outer_f1', 'outer_auc']].mean())
