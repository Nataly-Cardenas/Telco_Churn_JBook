"""Optimizaciones computacionales aplicadas sobre los modelos representativos.

Cada comparación parte de la configuración representativa de su familia (mejor F1
medio externo) y de los hiperparámetros que eligió la búsqueda oficial en cada
fold externo. Se entrena sobre el entrenamiento externo ya transformado y
balanceado por el mismo Pipeline oficial, y se evalúa en el fold reservado.
Así, "estándar" es exactamente lo que se ejecutó en el experimento y
"optimizado" es la implementación alternativa.
"""
from pathlib import Path
import cProfile
import io
import json
import os
import pstats
import threading
import time
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import cloudpickle
import numpy as np
import pandas as pd
import psutil
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OFICIAL = ROOT / 'resultados_modelado/oficial/20260920T030103_517c8c6b7906_69c6c7'
ANALISIS = ROOT / 'resultados_modelado/analisis_final'
SALIDA = ROOT / 'resultados_modelado/complementarios/optimizacion_computacional'
SEMILLA_GLOBAL = 42
REPRESENTANTES = {'knn': 'knn__smote__bayesian_tpe', 'naive_bayes': 'naive_bayes__none__random',
                  'logistic': 'logistic__smote__grid', 'random_forest': 'random_forest__smote__bayesian_tpe',
                  'xgboost': 'xgboost__smote__random', 'svm': 'svm__smote__genetic'}


class Medidor:
    """Mide tiempo de pared y pico de RSS (memoria del proceso, incluye código C) de un bloque."""

    def __init__(self, intervalo=0.002):
        self.intervalo = intervalo

    def __enter__(self):
        self.proceso = psutil.Process()
        self.base = self.pico = self.proceso.memory_info().rss
        self.activo = True
        self.hilo = threading.Thread(target=self._muestrear, daemon=True)
        self.hilo.start()
        self.inicio = time.perf_counter()
        return self

    def _muestrear(self):
        while self.activo:
            self.pico = max(self.pico, self.proceso.memory_info().rss)
            time.sleep(self.intervalo)

    def __exit__(self, *exc):
        self.segundos = time.perf_counter() - self.inicio
        self.activo = False
        self.hilo.join()
        self.pico = max(self.pico, self.proceso.memory_info().rss)
        self.mib = (self.pico - self.base) / 2 ** 20


def datos_desarrollo():
    """Variables originales y etiqueta de desarrollo en el orden oficial."""
    raw = pd.read_csv(ROOT / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    orden = pd.read_csv(ROOT / 'train_index.csv').iloc[:, 0].to_numpy()
    y = raw.Churn.map({'No': 0, 'Yes': 1}).astype(int).loc[orden]
    return raw.drop(columns=['customerID', 'Churn']).loc[orden], y


def unidad(eid, fold):
    """Fila oficial y Pipeline sin ajustar de una configuración en un fold de la realización 0."""
    unidades = pd.read_csv(ANALISIS / 'unidades.csv')
    fila = unidades[(unidades.experiment_id == eid) & (unidades.replicate == 0) &
                    (unidades.outer_fold == fold)].iloc[0]
    carpeta = OFICIAL / Path(str(fila.attempt_path).replace(chr(92), '/'))
    with (carpeta / 'pipeline.pkl').open('rb') as file:
        return fila, clone(cloudpickle.load(file))


def transformar_y_balancear(pipe, X_fit, y_fit):
    """Ajusta preprocesamiento y remuestreo del Pipeline oficial sobre el entrenamiento indicado."""
    prep = pipe[:-2]
    Zt = prep.fit_transform(X_fit, y_fit)
    Z, yz = pipe.named_steps['sampler'].fit_resample(Zt, y_fit)
    return prep, Z, yz


def matrices(eid, fold, X, y, splits):
    """Transforma y balancea con el Pipeline oficial; devuelve datos listos y el estimador original."""
    fila, pipe = unidad(eid, fold)
    split = splits[fold]
    X_fit, y_fit = X.iloc[split['fit_pos']], y.iloc[split['fit_pos']]
    X_eval, y_eval = X.iloc[split['eval_pos']], y.iloc[split['eval_pos']]
    prep, Z, yz = transformar_y_balancear(pipe, X_fit, y_fit)
    Ze = prep.transform(X_eval)
    estimador = pipe.named_steps['classifier'].estimator
    return (np.asarray(Z, dtype=np.float64), np.asarray(yz), np.asarray(Ze, dtype=np.float64),
            y_eval.to_numpy(), estimador, json.loads(fila.best_params), X_fit, y_fit, pipe)


def puntuar(modelo, Z):
    """Score continuo para AUC: probabilidad positiva o margen."""
    if hasattr(modelo, 'predict_proba'):
        return modelo.predict_proba(Z)[:, 1]
    return modelo.decision_function(Z)


def correr(nombre, variante, fabrica, Z, yz, Ze, ye, fold, referencia=None, repeticiones=5):
    """Entrena y predice una variante; devuelve tiempos, memoria, métricas y acuerdo con la estándar.

    Los tiempos y la memoria son la mediana de ``repeticiones`` ajustes y predicciones
    independientes; la inferencia mide solo ``predict`` sobre el fold externo completo.
    """
    medidas = []
    for _ in range(repeticiones):
        modelo = fabrica()
        with Medidor() as entrenamiento:
            modelo.fit(Z, yz)
        with Medidor() as inferencia:
            etiquetas = modelo.predict(Ze)
        medidas.append((entrenamiento.segundos, inferencia.segundos, entrenamiento.mib, inferencia.mib))
    medidas = np.median(np.array(medidas), axis=0)
    score = puntuar(modelo, Ze)
    fila = {'model': nombre, 'variant': variante, 'outer_fold': fold,
            'train_seconds': medidas[0], 'predict_seconds': medidas[1],
            'train_rss_mib': medidas[2], 'predict_rss_mib': medidas[3], 'repetitions': repeticiones,
            'f1': f1_score(ye, etiquetas), 'auc': roc_auc_score(ye, score),
            'agreement_with_standard': float(np.mean(etiquetas == referencia)) if referencia is not None else 1.0}
    if hasattr(modelo, 'best_iteration'):
        fila['trees_used'] = int(modelo.best_iteration) + 1
    return fila, etiquetas


def knn_faiss(params):
    """KNN aproximado con un grafo HNSW de FAISS (métrica L1, como la búsqueda oficial p=1)."""
    import faiss

    class KNNFaiss:
        def fit(self, Z, y):
            metrica = faiss.METRIC_L1 if params['p'] == 1 else faiss.METRIC_L2
            self.index = faiss.IndexHNSWFlat(Z.shape[1], 32, metrica)
            self.index.hnsw.efConstruction = 80
            self.index.hnsw.efSearch = max(2 * params['n_neighbors'], 64)
            self.index.add(np.ascontiguousarray(Z, dtype=np.float32))
            self.y = np.asarray(y)
            return self

        def predict_proba(self, Z):
            distancias, indices = self.index.search(np.ascontiguousarray(Z, dtype=np.float32), params['n_neighbors'])
            if params['weights'] == 'distance':
                d = np.sqrt(distancias) if params['p'] == 2 else distancias
                votos = 1.0 / np.maximum(d, 1e-12)
            else:
                votos = np.ones_like(distancias)
            positivos = (votos * (self.y[indices] == 1)).sum(axis=1)
            p1 = positivos / votos.sum(axis=1)
            return np.column_stack([1 - p1, p1])

        def predict(self, Z):
            return (self.predict_proba(Z)[:, 1] > 0.5).astype(int)
    return KNNFaiss()


def comparaciones_por_fold(fold, X, y, splits):
    """Variantes estándar vs optimizadas de cada familia en un fold externo."""
    filas = []
    # KNN: consulta exhaustiva frente a índices exactos y vecinos aproximados.
    Z, yz, Ze, ye, est, bp, *_ = matrices(REPRESENTANTES['knn'], fold, X, y, splits)
    base = dict(n_neighbors=bp['n_neighbors'], weights=bp['weights'], p=bp['p'], n_jobs=1)
    fila, ref = correr('knn', 'brute (estándar)', lambda: KNeighborsClassifier(algorithm='brute', **base), Z, yz, Ze, ye, fold)
    filas.append(fila)
    for alg in ['kd_tree', 'ball_tree']:
        filas.append(correr('knn', alg, lambda: KNeighborsClassifier(algorithm=alg, leaf_size=30, **base),
                            Z, yz, Ze, ye, fold, ref)[0])
    filas.append(correr('knn', 'FAISS HNSW (aprox.)', lambda: knn_faiss(bp), Z, yz, Ze, ye, fold, ref)[0])
    filas.append(correr('knn', 'brute, n_jobs=4', lambda: KNeighborsClassifier(algorithm='brute', **{**base, 'n_jobs': 4}),
                        Z, yz, Ze, ye, fold, ref)[0])

    # GaussianNB: ajuste completo frente a aprendizaje incremental por bloques.
    Z, yz, Ze, ye, est, bp, *_ = matrices(REPRESENTANTES['naive_bayes'], fold, X, y, splits)
    fila, ref = correr('naive_bayes', 'fit (estándar)', lambda: GaussianNB(var_smoothing=bp['var_smoothing']), Z, yz, Ze, ye, fold)
    filas.append(fila)

    class NBIncremental(GaussianNB):
        def fit(self, Z, y):
            for bloque in np.array_split(np.arange(len(Z)), 10):
                self.partial_fit(Z[bloque], y[bloque], classes=[0, 1])
            return self
    filas.append(correr('naive_bayes', 'partial_fit (10 bloques)', lambda: NBIncremental(var_smoothing=bp['var_smoothing']),
                        Z, yz, Ze, ye, fold, ref)[0])

    # Logística: liblinear (estándar) frente a SAGA, de coste lineal por época.
    Z, yz, Ze, ye, est, bp, *_ = matrices(REPRESENTANTES['logistic'], fold, X, y, splits)
    fila, ref = correr('logistic', 'liblinear (estándar)', lambda: clone(est), Z, yz, Ze, ye, fold)
    filas.append(fila)
    filas.append(correr('logistic', 'saga', lambda: LogisticRegression(C=bp['C'], penalty=bp['penalty'], solver='saga',
                        max_iter=2000, tol=1e-4, random_state=SEMILLA_GLOBAL), Z, yz, Ze, ye, fold, ref)[0])

    # Random Forest: paralelismo entre árboles (n_jobs) con el backend por hilos de scikit-learn.
    Z, yz, Ze, ye, est, bp, *_ = matrices(REPRESENTANTES['random_forest'], fold, X, y, splits)
    fila, ref = correr('random_forest', 'n_jobs=1 (estándar)', lambda: clone(est).set_params(n_jobs=1), Z, yz, Ze, ye, fold)
    filas.append(fila)
    for jobs in [2, 4, 8]:
        filas.append(correr('random_forest', f'n_jobs={jobs}', lambda: clone(est).set_params(n_jobs=jobs),
                            Z, yz, Ze, ye, fold, ref)[0])

    # XGBoost: histogramas (estándar) frente a método exacto y early stopping.
    Z, yz, Ze, ye, est, bp, X_fit, y_fit, pipe = matrices(REPRESENTANTES['xgboost'], fold, X, y, splits)
    fila, ref = correr('xgboost', 'hist (estándar)', lambda: clone(est), Z, yz, Ze, ye, fold)
    filas.append(fila)
    filas.append(correr('xgboost', 'exact', lambda: clone(est).set_params(tree_method='exact'), Z, yz, Ze, ye, fold, ref)[0])
    # Early stopping: el conjunto de validación sale del entrenamiento externo ANTES de SMOTE.
    a, b = train_test_split(np.arange(len(X_fit)), test_size=0.2, stratify=y_fit, random_state=SEMILLA_GLOBAL)
    pipe_es = clone(pipe)
    prep_es, Za, ya = transformar_y_balancear(pipe_es, X_fit.iloc[a], y_fit.iloc[a])
    Zv = prep_es.transform(X_fit.iloc[b])
    Za, Zv = np.asarray(Za, dtype=np.float64), np.asarray(Zv, dtype=np.float64)

    class XGBEarly(XGBClassifier):
        def fit(self, Z, y):
            return super().fit(Za, ya, eval_set=[(Zv, y_fit.iloc[b].to_numpy())], verbose=False)
    filas.append(correr('xgboost', 'hist, T=1000 sin early stopping', lambda: clone(est).set_params(n_estimators=1000),
                        Z, yz, Ze, ye, fold, ref)[0])
    filas.append(correr('xgboost', 'hist + early stopping', lambda: XGBEarly(**{**clone(est).get_params(),
                        'n_estimators': 1000, 'early_stopping_rounds': 30, 'eval_metric': 'logloss'}),
                        Z, yz, Ze, ye, fold, ref)[0])

    # SVM: núcleo RBF exacto (estándar) frente a alternativas lineales y aproximación de Nyström.
    Z, yz, Ze, ye, est, bp, *_ = matrices(REPRESENTANTES['svm'], fold, X, y, splits)
    fila, ref = correr('svm', 'SVC RBF (estándar)', lambda: clone(est), Z, yz, Ze, ye, fold)
    filas.append(fila)
    filas.append(correr('svm', 'LinearSVC', lambda: LinearSVC(C=bp['C'], dual='auto', max_iter=20000,
                        random_state=SEMILLA_GLOBAL), Z, yz, Ze, ye, fold, ref)[0])
    filas.append(correr('svm', 'SGD (hinge)', lambda: SGDClassifier(loss='hinge', alpha=1 / (bp['C'] * len(Z)),
                        max_iter=50, tol=1e-4, random_state=SEMILLA_GLOBAL), Z, yz, Ze, ye, fold, ref)[0])
    filas.append(correr('svm', 'Nyström(300) + LinearSVC', lambda: make_pipeline(
        Nystroem(gamma=bp['gamma'], n_components=300, random_state=SEMILLA_GLOBAL),
        LinearSVC(C=bp['C'], dual='auto', max_iter=5000, random_state=SEMILLA_GLOBAL)), Z, yz, Ze, ye, fold, ref)[0])
    return filas


def escalamiento(X, y, splits, repeticiones=3):
    """Tiempo frente a n (filas) y p (columnas) para estimar exponentes empíricos."""
    Z, yz, Ze, ye, *_ = matrices(REPRESENTANTES['svm'], 0, X, y, splits)
    rng = np.random.default_rng(SEMILLA_GLOBAL)
    consultas = Ze[:1000]
    fabricas = {
        'knn brute (predicción)': lambda: KNeighborsClassifier(41, p=1, algorithm='brute'),
        'knn kd_tree (predicción)': lambda: KNeighborsClassifier(41, p=1, algorithm='kd_tree'),
        'naive_bayes': lambda: GaussianNB(),
        'logistic liblinear': lambda: LogisticRegression(C=0.05, solver='liblinear'),
        'random_forest (100 árboles)': lambda: RandomForestClassifier(100, max_depth=8, n_jobs=1, random_state=SEMILLA_GLOBAL),
        'xgboost hist': lambda: XGBClassifier(n_estimators=100, max_depth=4, tree_method='hist', n_jobs=1, random_state=SEMILLA_GLOBAL),
        'xgboost exact': lambda: XGBClassifier(n_estimators=100, max_depth=4, tree_method='exact', n_jobs=1, random_state=SEMILLA_GLOBAL),
        'svm rbf': lambda: SVC(C=1.0, gamma=0.03),
        'svm LinearSVC': lambda: LinearSVC(C=0.1, dual='auto', max_iter=5000),
    }
    filas = []
    for eje, valores in [('n', [800, 1600, 3200, 6400]), ('p', [24, 48, 96, 192])]:
        for valor in valores:
            if eje == 'n':
                idx = rng.choice(len(Z), size=min(valor, len(Z)), replace=False)
                Zs, ys, Q = Z[idx], yz[idx], consultas
            else:
                idx = rng.choice(len(Z), size=3200, replace=False)
                ruido = rng.standard_normal((len(Z), valor - Z.shape[1]))
                ruido_q = rng.standard_normal((len(consultas), valor - Z.shape[1]))
                Zs, ys, Q = np.hstack([Z, ruido])[idx], yz[idx], np.hstack([consultas, ruido_q])
            for nombre, fabrica in fabricas.items():
                tiempos = []
                for _ in range(repeticiones):
                    modelo = fabrica()
                    inicio = time.perf_counter(); modelo.fit(Zs, ys); ajuste = time.perf_counter() - inicio
                    inicio = time.perf_counter(); modelo.predict(Q); prediccion = time.perf_counter() - inicio
                    tiempos.append(prediccion if 'predicción' in nombre else ajuste)
                filas.append({'axis': eje, 'value': valor, 'n': len(Zs), 'p': Zs.shape[1], 'model': nombre,
                              'median_seconds': float(np.median(tiempos)), 'repetitions': repeticiones})
    return pd.DataFrame(filas)


def paralelismo_cv(X, y, splits):
    """Validación cruzada de RF-SMOTE completa con distintos trabajadores y backends de joblib."""
    _, pipe = unidad(REPRESENTANTES['random_forest'], 0)
    filas = []
    referencia = None
    for backend, jobs in [('loky', 1), ('loky', 2), ('loky', 4), ('threading', 2), ('threading', 4)]:
        tiempos = []
        for _ in range(3):
            with parallel_backend(backend, n_jobs=jobs), Medidor() as medida:
                salida = cross_validate(clone(pipe), X, y, cv=[(np.asarray(s['fit_pos']), np.asarray(s['eval_pos']))
                                        for s in splits], scoring='f1', n_jobs=jobs)
            tiempos.append(medida.segundos)
        f1 = salida['test_score']
        referencia = f1 if referencia is None else referencia
        filas.append({'backend': backend, 'n_jobs': jobs, 'median_seconds': float(np.median(tiempos)),
                      'min_seconds': float(np.min(tiempos)), 'rss_mib_parent': medida.mib,
                      'f1_mean': float(f1.mean()), 'same_f1_as_serial': bool(np.allclose(f1, referencia))})
    tabla = pd.DataFrame(filas)
    serial = tabla.loc[(tabla.backend == 'loky') & (tabla.n_jobs == 1), 'median_seconds'].iloc[0]
    tabla['speedup'] = serial / tabla.median_seconds
    tabla['efficiency'] = tabla.speedup / tabla.n_jobs
    return tabla


def perfiles(X, y, splits):
    """cProfile del ajuste completo de SVM-SMOTE y de la predicción de KNN-SMOTE."""
    textos = {}
    for eid, accion in [(REPRESENTANTES['svm'], 'fit'), (REPRESENTANTES['knn'], 'predict')]:
        _, pipe = unidad(eid, 0)
        split = splits[0]
        X_fit, y_fit = X.iloc[split['fit_pos']], y.iloc[split['fit_pos']]
        X_eval = X.iloc[split['eval_pos']]
        if accion == 'predict':
            pipe.fit(X_fit, y_fit)
        perfil = cProfile.Profile()
        perfil.enable()
        pipe.fit(X_fit, y_fit) if accion == 'fit' else pipe.predict(X_eval)
        perfil.disable()
        buffer = io.StringIO()
        pstats.Stats(perfil, stream=buffer).sort_stats('cumulative').print_stats(18)
        textos[f'{eid}_{accion}'] = buffer.getvalue()
    return textos


def prueba_gpu():
    """Intenta XGBoost con CUDA y documenta el resultado sin inventar una aceleración."""
    Z = np.random.default_rng(0).standard_normal((200, 5)); yv = (Z[:, 0] > 0).astype(int)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        try:
            XGBClassifier(n_estimators=5, device='cuda').fit(Z, yv)
            estado = 'sin error'
        except Exception as error:  # el equipo no tiene GPU NVIDIA
            estado = f'error: {type(error).__name__}: {str(error)[:200]}'
    return {'device': 'cuda', 'estado': estado, 'avisos': [str(a.message)[:300] for a in avisos]}


if __name__ == '__main__':
    import sys
    X, y = datos_desarrollo()
    splits = json.loads((OFICIAL / 'splits.json').read_text(encoding='utf-8'))
    SALIDA.mkdir(parents=True, exist_ok=True)
    partes = sys.argv[1:] or ['comparaciones', 'escalamiento', 'paralelismo', 'perfiles', 'gpu']
    if 'comparaciones' in partes:
        tabla = pd.DataFrame([f for fold in range(5) for f in comparaciones_por_fold(fold, X, y, splits)])
        tabla.to_csv(SALIDA / 'comparaciones_por_fold.csv', index=False)
        print(tabla.groupby(['model', 'variant'], sort=False)[['train_seconds', 'predict_seconds', 'train_rss_mib',
              'f1', 'auc', 'agreement_with_standard']].mean())
    if 'escalamiento' in partes:
        escalamiento(X, y, splits).to_csv(SALIDA / 'escalamiento.csv', index=False)
    if 'paralelismo' in partes:
        tabla = paralelismo_cv(X, y, splits)
        tabla.to_csv(SALIDA / 'paralelismo_cv.csv', index=False)
        print(tabla)
    if 'perfiles' in partes:
        for nombre, texto in perfiles(X, y, splits).items():
            (SALIDA / f'perfil_{nombre}.txt').write_text(texto, encoding='utf-8')
    if 'gpu' in partes:
        resultado = prueba_gpu()
        (SALIDA / 'gpu.json').write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
        print(resultado)
