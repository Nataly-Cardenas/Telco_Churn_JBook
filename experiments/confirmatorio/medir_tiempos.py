"""Mide cuánto cuesta una evaluación de hiperparámetros por modelo y conjunto. Solo guarda segundos: no
calcula ninguna métrica de desempeño. Sirve para fijar el diseño antes del tag (sección 13 del protocolo)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import *  # noqa: E402,F403
import optuna  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.ERROR)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def medir(ds, nombre):
    X, y = cargar(ds)
    tr, _ = pliegues_externos(y, 0)[0]
    Xtr, ytr = X.iloc[tr], y[tr]
    a, b = pliegues_internos(ytr, 0, 0)[0]
    espacio = ESPACIOS["benchmarks"][nombre]["espacio"]
    s = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed_for("sampler", 7)))
    tiempos = []
    for _ in range(N):
        tri = s.ask()
        p = sugerir(tri, espacio)
        t0 = time.perf_counter()
        pipeline_calibrado(ds, nombre, p, 0).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])
        tiempos.append(time.perf_counter() - t0)
        s.tell(tri, 0.0)
    return {"ds": ds, "modelo": nombre, "seg_medio_un_pliegue_interno": float(np.mean(tiempos)), "seg_max": float(np.max(tiempos))}


if __name__ == "__main__":
    for d in DATASETS:          # crea la submuestra fija de KDD antes de lanzar procesos
        cargar(d)
    filas = Parallel(n_jobs=16)(delayed(medir)(d, m) for d in DATASETS for m in BENCHMARKS)
    df = pd.DataFrame(filas)
    df["horas_cpu_25_pliegues"] = df.seg_medio_un_pliegue_interno * 3 * 30 * 25 / 3600
    print(df.round(2).to_string())
    print(df.groupby("ds").horas_cpu_25_pliegues.sum().round(1))
    Path(__file__).with_name("tiempos_medidos.csv").write_text(df.to_csv(index=False), encoding="utf-8")
