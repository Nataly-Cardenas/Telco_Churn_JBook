"""Corridas únicas de Telco: cambio de distribución por tenure y hold-out final (sección 10 del protocolo).

Cada una repite el procedimiento completo de un pliegue externo (mismo presupuesto y mismos modelos):
  - tenure:  ajusta con tenure <= 24 y evalúa con tenure > 24, solo dentro del conjunto de desarrollo.
  - holdout: ajusta con los 5634 de desarrollo y evalúa una sola vez los 1409 del hold-out.

Uso: python experiments/confirmatorio/correr_finales.py tenure|holdout [n_procesos]
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import *  # noqa: E402,F403
from correr_anidado import guardar, ruta, tarea_benchmark, tarea_combinadores  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

# coordenadas de semilla reservadas: no coinciden con ninguna repetición de la validación anidada
COORD = {"tenure": (8, 0), "holdout": (9, 0)}


def datos(cual):
    X, y = cargar("telco")
    if cual == "tenure":
        ten = pd.to_numeric(X["tenure"]).to_numpy()
        return X, y, np.where(ten <= 24)[0], np.where(ten > 24)[0]
    Xh, yh = cargar_holdout()
    return pd.concat([X, Xh], ignore_index=True), np.r_[y, yh], np.arange(len(X)), np.arange(len(X), len(X) + len(Xh))


def un_modelo(cual, nombre):
    r, f = COORD[cual]
    destino = ruta(f"telco_{cual}", r, f, nombre)
    if destino.exists():
        return
    X, y, tr, te = datos(cual)
    guardar(destino, tarea_benchmark("telco", r, f, nombre, X, y, tr, te, BENCHMARKS.index(nombre)))
    print(cual, nombre, flush=True)


if __name__ == "__main__":
    cual = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    Parallel(n_jobs=n, batch_size=1)(delayed(un_modelo)(cual, m) for m in BENCHMARKS)
    r, f = COORD[cual]
    destino = ruta(f"telco_{cual}", r, f, "_combinadores")
    if not destino.exists():
        X, y, tr, te = datos(cual)
        exp = [pickle.load(open(ruta(f"telco_{cual}", r, f, m), "rb")) for m in BENCHMARKS]
        O, T = np.column_stack([e["oof"] for e in exp]), np.column_stack([e["pte"] for e in exp])
        pp = preprocesador("telco").fit(X.iloc[tr])
        guardar(destino, tarea_combinadores("telco", r, f, np.asarray(pp.transform(X.iloc[tr]), float), y[tr],
                                            np.asarray(pp.transform(X.iloc[te]), float), O, T))
    print("listo", cual, flush=True)
