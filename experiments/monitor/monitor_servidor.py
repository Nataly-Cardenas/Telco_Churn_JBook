"""Monitor de solo lectura de la corrida confirmatoria.

Garantías de diseño:
  - No escribe en el proyecto. Su única escritura es una caché en la carpeta temporal del sistema.
  - Cada checkpoint se lee una sola vez y se recuerda lo extraído. El resto de ciclos solo mira nombres y fechas.
  - El hold-out nunca se abre: de esos checkpoints solo se cuenta que existen.
  - Un único hilo consulta el disco cada 25 s. Los navegadores reciben esa foto en memoria, así que abrir más
    pestañas no aumenta las lecturas.
  - No importa numpy: un lector restringido de pickle ignora los arreglos y rechaza cualquier otra clase.
  - Corre con prioridad baja y escucha solo en 127.0.0.1, salvo que se pida --red-local.

Uso: python experiments/monitor/monitor_servidor.py [--puerto 8765] [--red-local]
"""
import argparse
import csv
import io
import json
import os
import pickle
import re
import socket
import statistics as st
import struct
import tempfile
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
RES = RAIZ / "experiments" / "resultados"
CRUDOS = RES / "crudos"
TABLAS = RES / "tablas"
ROBUSTEZ = RES / "robustez"
LOG = RES / "log_anidado.txt"
CACHE = Path(tempfile.gettempdir()) / "telco_monitor_cache.json"
PERIODO = 25
MAX_LECTURAS_POR_CICLO = 60     # la primera vez reparte la lectura en varios ciclos; luego llegan 1 o 2 por ciclo
GB = 2 ** 30

MODELOS = ["knn", "naive_bayes", "logistic", "decision_tree", "random_forest", "xgboost", "svm", "logistic_splines",
           "hist_gb", "lightgbm", "catboost"]
ABLACION = ["mapem_sin_prior", "mapem_K1", "mapem_regiones_duras", "promedio_simple", "stacking_logistico"]
CONJUNTOS = [
    {"id": "telco", "nombre": "Telco Churn", "reps": 5, "n_ajuste": 4507},
    {"id": "uci_563_iranian", "nombre": "Iranian Churn", "reps": 5, "n_ajuste": 2280},
    {"id": "openml_40701_churn", "nombre": "OpenML 40701", "reps": 5, "n_ajuste": 4000},
    {"id": "openml_1112_kddcup09_churn", "nombre": "KDDCup09", "reps": 2, "n_ajuste": 8000},
]
SCRIPTS = {"correr_anidado.py": "Validación anidada", "robustez.py": "Robustez y sensibilidad", "correr_finales.py": "Corrida final",
           "complementos_confirmatorio.py": "Complementos", "analizar.py": "Análisis estadístico"}
FACTOR_COMBINADORES = 1.25   # Telco: 25 tareas en 35 min con 10 procesos, frente a 670 s registrados por tarea
ESCENARIOS_ROBUSTEZ = 21
TABLAS_ANALISIS = ["metricas_por_pliegue.csv", "costos_por_pliegue.csv", "friedman_telco.json", "logloss_medio_por_conjunto.csv",
                   "familia_resultados.csv", "bayesiana_rope.csv", "resumen_modelos.csv", "veredicto_R3.json"]
CAPITULOS = {"trazabilidad": "Trazabilidad", "literatura": "Literatura y novedad", "validacion": "Validación estadística",
             "evaluacion": "Evaluación y robustez", "dashboard": "Tablero y despliegue", "conclusion": "Conclusiones",
             "reproducib": "Reproducibilidad"}
DASH = Path.home() / "Documents" / "Dash_Telco_Churn"


# ---------------------------------------------------------------------------------- lectura segura
class _Ignorado:
    """Sustituye a los arreglos de numpy: acepta su estado y lo descarta."""

    def __init__(self, *a, **k):
        self.args = a

    def __setstate__(self, estado):
        pass


def _escalar(dtipo, datos=None):
    codigo = dtipo.args[0] if getattr(dtipo, "args", None) else ""
    if not isinstance(datos, (bytes, bytearray)):
        return None
    if codigo in ("f8", "float64"):
        return struct.unpack("<d", datos)[0]
    if codigo in ("f4", "float32"):
        return struct.unpack("<f", datos)[0]
    if codigo in ("i8", "int64"):
        return struct.unpack("<q", datos)[0]
    if codigo in ("i4", "int32"):
        return struct.unpack("<i", datos)[0]
    if codigo in ("b1", "bool"):
        return bool(datos[0])
    return None


class LectorSeguro(pickle.Unpickler):
    """Solo deja pasar los constructores de numpy, y sin importarlo. Nada más se puede instanciar."""

    def find_class(self, modulo, nombre):
        if modulo.split(".")[0] == "numpy":
            if nombre == "scalar":
                return _escalar
            if nombre in ("_reconstruct", "ndarray", "dtype", "_frombuffer"):
                return _Ignorado
        raise pickle.UnpicklingError(f"clase no permitida: {modulo}.{nombre}")


def leer_pickle(ruta):
    with open(ruta, "rb") as fh:
        datos = fh.read()
    return LectorSeguro(io.BytesIO(datos)).load()


def _mediana(valores):
    v = [x for x in valores if isinstance(x, (int, float))]
    return st.median(v) if v else None


def extraer_benchmark(o):
    reg = o.get("registro") or []
    return {"tipo": "b", "busq": o["seg_busqueda"], "ajuste": o["seg_ajuste"], "pred": o["seg_pred"], "n_pred": o["n_pred"],
            "ensayos": o["ensayos"], "fallidos": o["fallidos"], "trial_med": _mediana([x.get("seg") for x in reg]),
            "ll_interno": o.get("valor_interno") if isinstance(o.get("valor_interno"), (int, float)) else None}


def extraer_combinadores(o):
    R = o["RegionalMAPEM"]
    reg = R.get("registro") or []
    valores = [x["valor"] for x in reg if isinstance(x.get("valor"), (int, float))]
    return {"tipo": "c", "busq": R["seg_busqueda"], "ajuste": R["seg_ajuste"], "pred": R["seg_pred"], "n_pred": R["n_pred"],
            "ensayos": R["ensayos"], "fallidos": R["fallidos"], "trial_med": _mediana([x.get("seg") for x in reg]),
            "ll_interno": min(valores) if valores else None, "n_iter": R.get("n_iter"), "convergio": R.get("convergio"),
            "ablacion": [a for a in ABLACION if a in o]}


# ---------------------------------------------------------------------------------- recorrido del disco
def recorrer(base, profundidad):
    """Devuelve (partes relativas, fecha de modificación, tamaño) de los .pkl, sin abrirlos."""
    salida = []

    def visitar(d, partes):
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        if len(partes) < profundidad:
                            visitar(e.path, partes + [e.name])
                    elif e.name.endswith(".pkl"):
                        s = e.stat()
                        salida.append((partes + [e.name[:-4]], s.st_mtime, s.st_size))
        except OSError:
            pass

    visitar(base, [])
    return salida


class LectorLog:
    """Lee solo lo nuevo del registro de la corrida, a partir de la última posición conocida."""

    def __init__(self):
        self.pos = 0
        self.intentos = []
        self.error_tras_intento = None
        self.terminados = {}
        self.ultima_linea = ""
        self.fin_correcto = False

    def actualizar(self):
        try:
            t = LOG.stat().st_size
            if t <= self.pos:
                return
            with open(LOG, "rb") as fh:
                fh.seek(self.pos)
                nuevos = fh.read()
            # no consumir una línea a medio escribir
            corte = nuevos.rfind(b"\n")
            if corte < 0:
                return
            self.pos += corte + 1
            for linea in nuevos[:corte].decode("utf-8", "replace").splitlines():
                if linea.startswith("[I 20"):
                    continue
                m = re.match(r"=== intento (\d+) .*? (\d+:\d+:\d+) .*?(\d{4}) con (\d+) procesos", linea)
                if m:
                    self.intentos.append({"n": int(m.group(1)), "linea": linea, "procesos": int(m.group(4)), "reloj": m.group(2)})
                    self.error_tras_intento = None
                    continue
                m = re.match(r"== (\S+) terminado en ([\d.]+) h", linea)
                if m:
                    self.terminados[m.group(1)] = float(m.group(2))
                    continue
                if re.search(r"(Traceback|WinError|MemoryError)", linea):
                    self.error_tras_intento = linea.strip()[:200] if "Traceback" not in linea else (self.error_tras_intento or "Traceback")
                if linea.strip():
                    self.ultima_linea = linea.strip()[:160]
        except OSError:
            pass


def contar_filas(ruta):
    """Filas de datos de un CSV. Cuenta saltos de línea; no interpreta el contenido."""
    try:
        with open(ruta, "rb") as fh:
            return max(0, sum(b.count(b"\n") for b in iter(lambda: fh.read(65536), b"")) - 1)
    except OSError:
        return None


def existe(p):
    try:
        return Path(p).exists()
    except OSError:
        return False


def edad_archivo(p):
    try:
        return os.stat(p).st_mtime
    except OSError:
        return None


def seccion_12_con_entradas():
    try:
        t = (RAIZ / "experiments" / "protocolo.md").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"## 12\. Desviaciones(.*?)## 13\.", t, re.S)
    return bool(m and re.search(r"20\d\d-\d\d-\d\d", m.group(1)))


# ---------------------------------------------------------------------------------- monitor
class Monitor:
    def __init__(self):
        self.foto = {"estado_general": {"codigo": "iniciando", "detalle": "Primera lectura en curso"}}
        self.candado = threading.Lock()
        self.datos = {}
        self.fallos_lectura = {}
        self.log = LectorLog()
        self.medido = self._cargar_medicion()
        self.proc_propio = psutil.Process()
        self.cache_cambio = False
        self._cargar_cache()
        psutil.cpu_percent(None)
        try:
            self.proc_propio.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        except (psutil.Error, OSError, AttributeError):
            pass

    def _cargar_medicion(self):
        m = {}
        try:
            with open(RAIZ / "experiments" / "confirmatorio" / "tiempos_medidos.csv", encoding="utf-8", newline="") as fh:
                for f in csv.DictReader(fh):
                    m[(f["ds"], f["modelo"])] = float(f["seg_medio_un_pliegue_interno"]) * 90
        except OSError:
            pass
        return m

    def _cargar_cache(self):
        try:
            self.datos = json.loads(CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.datos = {}

    def _guardar_cache(self):
        if not self.cache_cambio:
            return
        try:
            tmp = CACHE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.datos), encoding="utf-8")
            tmp.replace(CACHE)
            self.cache_cambio = False
        except OSError:
            pass

    # ---- procesos
    def procesos(self):
        info = {"principal": None, "trabajadores": 0, "reintentos": False, "script": None, "rss_mb": 0.0, "pid": None}
        try:
            pr = []
            for p in psutil.process_iter(["pid", "ppid", "name"]):
                n = (p.info["name"] or "").lower()
                if n.startswith("python") or n in ("bash.exe", "sh.exe", "bash", "sh"):
                    if p.info["pid"] != self.proc_propio.pid:
                        pr.append(p)
            cmd, args = {}, {}
            for p in pr:
                try:
                    a = p.cmdline()
                except psutil.Error:
                    a = []
                args[p.info["pid"]] = a
                cmd[p.info["pid"]] = " ".join(a)
            for p in pr:
                # solo cuenta el script que se ejecuta, no un comando que lo mencione
                ejecuta = {re.split(r"[\\/]", x)[-1] for x in args[p.info["pid"]][1:3]}
                if "correr_con_reintentos.sh" in ejecuta:
                    info["reintentos"] = True
                if (p.info["name"] or "").lower().startswith("python") and info["principal"] is None:
                    for s, etiqueta in SCRIPTS.items():
                        if s in ejecuta:
                            info.update(principal=etiqueta, script=s, pid=p.info["pid"])
            if info["pid"]:
                for p in pr:
                    if p.info["ppid"] == info["pid"] and "resource_tracker" not in cmd[p.info["pid"]]:
                        info["trabajadores"] += 1
                        try:
                            info["rss_mb"] += p.memory_info().rss / 1e6
                        except psutil.Error:
                            pass
        except psutil.Error:
            info["error"] = True
        return info

    # ---- checkpoints
    def actualizar_checkpoints(self):
        hallados = {}
        leidos = 0
        for partes, mt, tam in recorrer(CRUDOS, 2):
            if len(partes) != 3:
                continue
            ds, celda, nombre = partes
            clave = f"{ds}/{celda}/{nombre}"
            hallados[clave] = (mt, tam)
            if ds == "telco_holdout":           # el hold-out nunca se abre
                continue
            prev = self.datos.get(clave)
            if prev and prev.get("mt") == mt and prev.get("tam") == tam:
                continue
            if leidos >= MAX_LECTURAS_POR_CICLO or self.fallos_lectura.get(clave, 0) >= 5:
                continue
            try:
                o = leer_pickle(CRUDOS / ds / celda / f"{nombre}.pkl")
                d = extraer_combinadores(o) if nombre == "_combinadores" else extraer_benchmark(o)
                d.update(mt=mt, tam=tam)
                self.datos[clave] = d
                self.cache_cambio = True
                self.fallos_lectura.pop(clave, None)
            except Exception:                   # archivo en uso o incompleto: se reintenta en el siguiente ciclo
                self.fallos_lectura[clave] = self.fallos_lectura.get(clave, 0) + 1
            leidos += 1
            time.sleep(0.003)
        return hallados, leidos

    # ---- estimación
    def seg_tarea(self, d):
        s = d["busq"] + d["ajuste"] + d["pred"]
        return s * FACTOR_COMBINADORES if d["tipo"] == "c" else s

    def estimador(self, ds_id, hechos, inicio_intento):
        """Devuelve, por modelo, la duración esperada (baja, central, alta) y si viene de tiempos reales."""
        por_ds = {}
        for (ds, modelo), lista in hechos.items():
            por_ds.setdefault(ds, {})[modelo] = lista
        def media(lista):
            recientes = [x for x in lista if inicio_intento and x["mt"] >= inicio_intento]
            base = recientes if len(recientes) >= 3 else lista
            return sum(self.seg_tarea(x) for x in base) / len(base)
        salida = {}
        for m in MODELOS:
            lista = por_ds.get(ds_id, {}).get(m, [])
            if len(lista) >= 3:
                e = media(lista)
                salida[m] = (0.85 * e, e, 1.15 * e, True)
                continue
            base = self.medido.get((ds_id, m))
            if base is None:
                salida[m] = (None, None, None, False)
                continue
            fac = []
            for ds2, modelos in por_ds.items():
                if len(modelos.get(m, [])) >= 3 and (ds2, m) in self.medido:
                    fac.append(media(modelos[m]) / self.medido[(ds2, m)])
            if fac:
                salida[m] = (base * min(fac), base * st.median(fac), base * max(max(fac), 1.0), False)
            else:
                salida[m] = (base * 0.4, base, base * 1.2, False)
        comb = por_ds.get(ds_id, {}).get("_combinadores", [])
        if len(comb) >= 3:
            e = media(comb)
            salida["_combinadores"] = (0.85 * e, e, 1.15 * e, True)
        else:
            ref = por_ds.get("telco", {}).get("_combinadores", [])
            if ref:
                n = next(c["n_ajuste"] for c in CONJUNTOS if c["id"] == ds_id)
                e = media(ref) * n / 4507
                salida["_combinadores"] = (0.6 * e, e, 2.0 * e, False)
            else:
                salida["_combinadores"] = (None, None, None, False)
        return salida

    # ---- ciclo principal
    def ciclo(self):
        t0 = time.perf_counter()
        ahora = time.time()
        self.log.actualizar()
        hallados, leidos = self.actualizar_checkpoints()
        pr = self.procesos()

        ultimo_intento = self.log.intentos[-1] if self.log.intentos else None
        inicio_intento = None
        if ultimo_intento:
            try:
                h, mi, se = map(int, ultimo_intento["reloj"].split(":"))
                m = re.search(r"(\w{3}) (\w{3}) +(\d+) ", ultimo_intento["linea"])
                meses = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
                ano = int(re.search(r"(\d{4}) con", ultimo_intento["linea"]).group(1))
                inicio_intento = datetime(ano, meses[m.group(2)], int(m.group(3)), h, mi, se).timestamp() if m else None
            except (ValueError, KeyError, AttributeError):
                inicio_intento = None

        # tareas hechas
        hechos = {}
        for clave, d in self.datos.items():
            ds, celda, nombre = clave.split("/")
            if clave not in hallados or ds not in {c["id"] for c in CONJUNTOS}:
                continue
            hechos.setdefault((ds, nombre), []).append(d)

        n_trab = pr["trabajadores"]
        n_conf = ultimo_intento["procesos"] if ultimo_intento else 10
        n_calc = n_trab or n_conf

        # planificación esperada: conjunto, etapa y cola
        activos, conjuntos_salida, celdas = [], [], {}
        ds_activo, etapa_activa = None, None
        fin_ultimo_ckpt = max((v[0] for v in hallados.values()), default=None)
        anidada_viva = pr["script"] == "correr_anidado.py"
        estimadores = {}
        for c in CONJUNTOS:
            cola_b = [(r, f, m) for r in range(c["reps"]) for f in range(5) for m in MODELOS]
            cola_c = [(r, f, "_combinadores") for r in range(c["reps"]) for f in range(5)]
            pend_b = [t for t in cola_b if f"{c['id']}/r{t[0]}_f{t[1]}/{t[2]}" not in hallados]
            pend_c = [t for t in cola_c if f"{c['id']}/r{t[0]}_f{t[1]}/{t[2]}" not in hallados]
            if ds_activo is None and (pend_b or pend_c):
                ds_activo = c["id"]
                etapa_activa = "benchmarks" if pend_b else "combinadores"
                if anidada_viva:
                    cola = cola_b if pend_b else cola_c
                    pend = pend_b or pend_c
                    fines = sorted(hallados[f"{c['id']}/r{t[0]}_f{t[1]}/{t[2]}"][0] for t in cola
                                   if f"{c['id']}/r{t[0]}_f{t[1]}/{t[2]}" in hallados
                                   and (not inicio_intento or hallados[f"{c['id']}/r{t[0]}_f{t[1]}/{t[2]}"][0] >= inicio_intento))
                    previos = [v[0] for k, v in hallados.items() if not (k.startswith(c["id"] + "/") and (etapa_activa == "benchmarks" or k.endswith("_combinadores")))]
                    inicio_etapa = max([inicio_intento or 0] + [x for x in previos if not fines or x <= fines[0]])
                    for i, t in enumerate(pend[:n_trab]):
                        # con cola FIFO, la tarea que ocupa el puesto q empezó al terminar la tarea q - N
                        q = len(fines) + i
                        idx = q - n_trab
                        inicio = fines[idx] if idx >= 0 else inicio_etapa
                        activos.append({"ds": c["id"], "ds_nombre": c["nombre"], "r": t[0], "f": t[1], "modelo": t[2], "desde": inicio})

        act_clave = {(a["ds"], a["r"], a["f"], a["modelo"]): a for a in activos}

        # ventana de utilización observada
        ventana = 5400
        desde = max(ahora - ventana, inicio_intento or 0)
        dur = max(ahora - desde, 1)
        ocupado = 0.0
        for lista in hechos.values():
            ocupado += sum(self.seg_tarea(x) for x in lista if x["mt"] >= desde)
        observados = ocupado / dur
        efectivos = min(n_calc, max(observados, 0.6 * n_calc))

        # recorrer conjuntos: estado, celdas y estimación secuencial
        acumulado = {"bajo": 0.0, "medio": 0.0, "alto": 0.0}
        hecho_trabajo = 0.0
        restante_trabajo_medio = 0.0
        hay_sin_estimar = False
        fiabilidad_global = "alta"
        for c in CONJUNTOS:
            cid = c["id"]
            est = self.estimador(cid, hechos, inicio_intento)
            estimadores[cid] = est
            matrices, listos, pendientes = [], [], []
            trabajo = [0.0, 0.0, 0.0]
            trabajo_real = 0.0
            total, hechas = 0, 0
            seg_proceso = 0.0
            fechas = []
            faltan_estim = False
            for m in MODELOS + ["_combinadores"]:
                filas = []
                n_h, seg_m, fallos_ev = 0, 0.0, 0
                e = est[m]
                for r in range(c["reps"]):
                    fila = []
                    for f in range(5):
                        clave = f"{cid}/r{r}_f{f}/{m}"
                        total += 1
                        en_disco = clave in hallados
                        d = self.datos.get(clave) if en_disco else None
                        a = act_clave.get((cid, r, f, m))
                        celda = {"r": r, "f": f, "ckpt": f"experiments/resultados/crudos/{cid}/r{r}_f{f}/{m}.pkl"}
                        if en_disco:
                            hechas += 1
                            n_h += 1
                            fechas.append(hallados[clave][0])
                            if d:
                                s = self.seg_tarea(d)
                                seg_proceso += s
                                seg_m += d["busq"] + d["ajuste"] + d["pred"]
                                fallos_ev += d["fallidos"] or 0
                                celda.update(e="ok" if not d["fallidos"] else "fallas", seg=d["busq"] + d["ajuste"] + d["pred"], ev=d["ensayos"], fa=d["fallidos"],
                                             fin=d["mt"], ll=d.get("ll_interno"))
                                if d["tipo"] == "c":
                                    celda.update(iter=d.get("n_iter"), conv=d.get("convergio"), abl=len(d.get("ablacion") or []))
                            else:
                                celda.update(e="leyendo" if self.fallos_lectura.get(clave, 0) < 5 else "ilegible", fin=hallados[clave][0])
                        else:
                            celda["e"] = "activa" if a else "pendiente"
                            if a:
                                celda["desde"] = a["desde"]
                            if e[1] is None:
                                faltan_estim = True
                            else:
                                resto = 1.0
                                if a:
                                    transc = ahora - a["desde"]
                                    trabajo[0] += max(e[0] - transc, 0.1 * e[0])
                                    trabajo[1] += max(e[1] - transc, 0.1 * e[1])
                                    trabajo[2] += max(e[2] - transc, 0.1 * e[2])
                                    resto = 0.0
                                trabajo[0] += resto * e[0]
                                trabajo[1] += resto * e[1]
                                trabajo[2] += resto * e[2]
                                if e[3]:
                                    trabajo_real += resto * e[1]
                        fila.append(celda)
                    filas.append(fila)
                tot_m = c["reps"] * 5
                nombre = "RegionalMAPEM y combinadores" if m == "_combinadores" else m
                matrices.append({"modelo": m, "nombre": nombre, "hechas": n_h, "total": tot_m, "filas": filas, "seg_medio": seg_m / n_h if n_h else None,
                                 "horas": seg_m / 3600, "fallos": fallos_ev})
                if m != "_combinadores":
                    (listos if n_h == tot_m else pendientes).append(m)
            hecho_trabajo += seg_proceso
            restante_trabajo_medio += trabajo[1]
            completo = hechas == total
            if completo:
                estado = "completado"
            elif cid == ds_activo and pr["script"] == "correr_anidado.py":
                estado = "en_ejecucion"
            elif hechas:
                estado = "detenido" if not pr["reintentos"] else "en_ejecucion"
            else:
                estado = "pendiente"
            if self.log.error_tras_intento and cid == ds_activo and not pr["principal"] and not pr["reintentos"]:
                estado = "error"
            fiab = None
            rest = None
            fin = None
            if not completo:
                if trabajo[1] > 0:
                    fr = trabajo_real / trabajo[1]
                    fiab = "alta" if fr >= 0.7 else ("media" if fr > 0 else "baja")
                    if fiab == "baja" or fiabilidad_global == "alta" or (fiab == "media" and fiabilidad_global != "baja"):
                        fiabilidad_global = fiab if fiab != "alta" else fiabilidad_global
                acumulado["bajo"] += trabajo[0] / n_calc
                acumulado["medio"] += trabajo[1] / max(efectivos, 1)
                acumulado["alto"] += trabajo[2] / max(efectivos, 1)
                rest = {"bajo": trabajo[0] / n_calc, "medio": trabajo[1] / max(efectivos, 1), "alto": trabajo[2] / max(efectivos, 1)}
                fin = {"bajo": ahora + acumulado["bajo"], "medio": ahora + acumulado["medio"], "alto": ahora + acumulado["alto"]}
            hay_sin_estimar = hay_sin_estimar or faltan_estim
            conjuntos_salida.append({"id": cid, "nombre": c["nombre"], "estado": estado, "hechas": hechas, "total": total, "pct": 100 * hechas / total,
                                     "horas_proceso": seg_proceso / 3600, "reloj": (max(fechas) - min(fechas)) if len(fechas) > 1 else 0, "inicio": min(fechas) if fechas else None,
                                     "ult": max(fechas) if fechas else None, "modelos_listos": listos, "modelos_pendientes": pendientes,
                                     "restante": rest, "fin": fin, "fiabilidad": fiab, "reps": c["reps"], "registro_h": self.log.terminados.get(cid)})
            celdas[cid] = matrices

        anidada_total = sum(x["total"] for x in conjuntos_salida)
        anidada_hechas = sum(x["hechas"] for x in conjuntos_salida)
        anidada_completa = anidada_total == anidada_hechas
        computo_pct = None
        if restante_trabajo_medio + hecho_trabajo > 0:
            computo_pct = 100 * hecho_trabajo / (restante_trabajo_medio + hecho_trabajo)

        etapas, posteriores = self.etapas(hallados, conjuntos_salida, pr, hechos, n_calc, anidada_completa)
        fin_anidada = None if anidada_completa else {k: ahora + v for k, v in acumulado.items()}
        total_rest = {"bajo": acumulado["bajo"] + posteriores[0], "medio": acumulado["medio"] + posteriores[1], "alto": acumulado["alto"] + posteriores[2]}
        fin_todo = {k: ahora + v for k, v in total_rest.items()}

        # estado general
        compl_computo = anidada_completa and all(e["estado"] == "completado" for e in etapas if e["clase"] == "computo")
        if pr["principal"]:
            cod, det = "ejecutando", pr["principal"]
        elif pr["reintentos"]:
            cod, det = "ejecutando", "Reinicio automático en espera tras una interrupción"
        elif compl_computo:
            cod, det = "completado", "Todas las etapas de cómputo tienen evidencia de cierre"
        elif self.log.error_tras_intento:
            cod, det = "error", self.log.error_tras_intento
        elif anidada_completa:
            cod, det = "detenido", "Validación anidada terminada. La siguiente etapa no se ha lanzado"
        else:
            cod, det = "detenido", "No hay ningún proceso de experimento activo"

        mem = psutil.virtual_memory()
        try:
            mem_propia = self.proc_propio.memory_info().rss / 1e6
            cpu_propia = sum(self.proc_propio.cpu_times()[:2])
        except psutil.Error:
            mem_propia, cpu_propia = None, None
        try:
            inicio_log = os.stat(LOG).st_ctime
        except OSError:
            inicio_log = None
        alarmas = []
        if ds_activo and existe(CRUDOS / "telco_holdout") and not anidada_completa:
            alarmas.append("Hay checkpoints de hold-out antes de terminar la validación anidada.")
        if mem.available < 0.3 * GB:
            alarmas.append(f"RAM disponible muy baja: {mem.available / GB:.2f} GB.")
        ilegibles = sum(1 for v in self.fallos_lectura.values() if v >= 5)
        if ilegibles:
            alarmas.append(f"{ilegibles} checkpoints no se pudieron leer tras 5 intentos.")
        if pr.get("trabajadores") and ultimo_intento and pr["trabajadores"] != ultimo_intento["procesos"] and anidada_viva:
            alarmas.append(f"Se configuraron {ultimo_intento['procesos']} procesos y hay {pr['trabajadores']} activos.")

        nombre_activo = next((c["nombre"] for c in CONJUNTOS if c["id"] == ds_activo), None)
        foto = {
            "ahora": ahora,
            "periodo": PERIODO,
            "estado_general": {"codigo": cod, "detalle": det},
            "actual": {"conjunto": nombre_activo if anidada_viva else None, "etapa": etapa_activa if anidada_viva else None, "proceso": pr["principal"]},
            "tareas": {"hechas": anidada_hechas, "total": anidada_total, "pct": 100 * anidada_hechas / anidada_total, "pct_computo": computo_pct},
            "tiempos": {"inicio": inicio_log, "inicio_intento": inicio_intento, "ultimo_ckpt": fin_ultimo_ckpt, "intento": ultimo_intento["n"] if ultimo_intento else None},
            "eta": {"anidada": fin_anidada, "todo": fin_todo, "restante_anidada": acumulado, "restante_todo": total_rest, "fiabilidad": fiabilidad_global,
                    "efectivos": efectivos, "observados": observados, "n_procesos": n_calc, "sin_estimar": hay_sin_estimar,
                    "por_leer": sum(1 for k in hallados if k not in self.datos and not k.startswith("telco_holdout/"))},
            "sistema": {"ram_disp_gb": mem.available / GB, "ram_total_gb": mem.total / GB, "cpu_pct": psutil.cpu_percent(None), "procesos": n_trab,
                        "rss_trabajadores_gb": pr["rss_mb"] * 1e6 / GB if pr["rss_mb"] else None, "script": pr["script"]},
            "conjuntos": conjuntos_salida,
            "celdas": celdas,
            "activos": activos,
            "etapas": etapas,
            "alarmas": alarmas,
            "monitor": {"ms_ciclo": (time.perf_counter() - t0) * 1000, "lecturas_nuevas": leidos, "ram_mb": mem_propia, "cpu_seg": cpu_propia, "en_cache": len(self.datos), "periodo": PERIODO},
        }
        self._guardar_cache()
        with self.candado:
            self.foto = foto

    # ---- otras etapas: solo se dan por terminadas con evidencia comprobable
    def etapas(self, hallados, conjuntos, pr, hechos, n_calc, anidada_completa):
        salida = []
        def agrega(id_, nombre, clase, estado, progreso, evidencia, nota=None):
            salida.append({"id": id_, "nombre": nombre, "clase": clase, "estado": estado, "progreso": progreso, "evidencia": evidencia, "nota": nota})

        # combinadores y ablación salen de los checkpoints _combinadores
        h_c = [k for k in hallados if k.endswith("/_combinadores") and k.split("/")[0] in {c["id"] for c in CONJUNTOS}]
        total_c = sum(c["reps"] * 5 for c in CONJUNTOS)
        activo = pr["script"] == "correr_anidado.py" and any(x["estado"] == "en_ejecucion" for x in conjuntos)
        agrega("combinadores", "Combinadores y RegionalMAPEM", "computo", "completado" if len(h_c) == total_c else ("en_curso" if h_c else "pendiente"),
               {"hechas": len(h_c), "total": total_c}, f"{len(h_c)} checkpoints _combinadores.pkl con su búsqueda registrada")
        abl = sum(len(self.datos.get(k, {}).get("ablacion") or []) for k in h_c)
        agrega("ablacion", "Estudios de ablación", "computo", "completado" if abl == total_c * 5 else ("en_curso" if abl else "pendiente"),
               {"hechas": abl, "total": total_c * 5}, "Variantes presentes dentro de _combinadores.pkl: sin prior, K = 1, regiones duras, promedio y stacking")

        # robustez
        rob = recorrer(ROBUSTEZ, 2)
        modelos_rob = sum(1 for p, _, _ in rob if p[-1] != "_resumen")
        resumen_rob = sum(1 for p, _, _ in rob if p[-1] == "_resumen")
        tot_m_r, tot_r_r = ESCENARIOS_ROBUSTEZ * 5 * 11, ESCENARIOS_ROBUSTEZ * 5
        filas_sens = contar_filas(ROBUSTEZ / "sensibilidad_K_c.csv") if existe(ROBUSTEZ / "sensibilidad_K_c.csv") else None
        hechas_r = modelos_rob + resumen_rob
        corriendo = pr["script"] == "robustez.py"
        est_r = "completado" if hechas_r == tot_m_r + tot_r_r and filas_sens == 1600 else ("en_curso" if corriendo else ("parcial" if hechas_r else "pendiente"))
        agrega("robustez", "Robustez y sensibilidad", "computo", est_r, {"hechas": hechas_r, "total": tot_m_r + tot_r_r},
               f"{modelos_rob} de {tot_m_r} ajustes, {resumen_rob} de {tot_r_r} resúmenes; sensibilidad: {filas_sens if filas_sens is not None else 'sin archivo'} de 1600 filas")

        def final(cual, nombre, r, nota=None):
            h = [k for k in hallados if k.startswith(f"telco_{cual}/r{r}_f0/")]
            est = "completado" if len(h) == 12 else ("en_curso" if pr["script"] == "correr_finales.py" else ("parcial" if h else "pendiente"))
            agrega(cual, nombre, "computo", est, {"hechas": len(h), "total": 12}, f"{len(h)} de 12 checkpoints (11 modelos y combinadores)", nota)
        final("tenure", "Evaluación de tenure", 8)
        final("holdout", "Hold-out final", 9, "El monitor solo cuenta estos archivos y nunca los abre. No muestra valores del hold-out.")

        # análisis: se comprueba la señal de cierre sin exponer ningún resultado
        fam = TABLAS / "familia_resultados.csv"
        calc = None
        if existe(fam):
            try:
                with open(fam, encoding="utf-8", newline="") as fh:
                    calc = sum(1 for f in csv.DictReader(fh) if f.get("calculada") == "True")
            except (OSError, csv.Error):
                calc = None
        ver = TABLAS / "veredicto_R3.json"
        parcial = None
        if existe(ver):
            try:
                parcial = json.loads(ver.read_text(encoding="utf-8")).get("parcial")
            except (OSError, ValueError):
                parcial = None
        ho_mt = edad_archivo(CRUDOS / "telco_holdout" / "r9_f0" / "_combinadores.pkl")
        ver_mt = edad_archivo(ver)
        analisis_listo = anidada_completa and parcial is False and calc == 223 and ho_mt is not None and ver_mt is not None and ver_mt > ho_mt
        analiza = pr["script"] == "analizar.py"
        agrega("estadistica", "Pruebas estadísticas", "computo", "completado" if analisis_listo else ("en_curso" if analiza else ("parcial" if calc else "pendiente")),
               {"hechas": calc or 0, "total": 223}, f"Comparaciones calculadas: {calc if calc is not None else 'sin archivo'} de 223. Solo se cuenta la columna de control, no los p ni los efectos.",
               "Cierra solo con 223 de 223, veredicto no parcial y fecha posterior al hold-out.")
        n_tab = sum(1 for t in TABLAS_ANALISIS if existe(TABLAS / t))
        agrega("tablas", "Generación de resultados y tablas", "computo", "completado" if analisis_listo and n_tab == len(TABLAS_ANALISIS) else ("en_curso" if analiza else ("parcial" if n_tab else "pendiente")),
               {"hechas": n_tab, "total": len(TABLAS_ANALISIS)}, f"{n_tab} de {len(TABLAS_ANALISIS)} tablas en experiments/resultados/tablas")

        compl = pr["script"] == "complementos_confirmatorio.py"
        fd = contar_filas(TABLAS / "combinaciones_dashboard.csv") if existe(TABLAS / "combinaciones_dashboard.csv") else None
        agrega("comb_dash", "Combinaciones para el tablero predictivo", "computo", "completado" if fd == 175 else ("en_curso" if compl else ("parcial" if fd else "pendiente")),
               {"hechas": fd or 0, "total": 175}, f"combinaciones_dashboard.csv: {fd if fd is not None else 'sin archivo'} de 175 filas")
        fe = contar_filas(TABLAS / "escalabilidad.csv") if existe(TABLAS / "escalabilidad.csv") else None
        fm = contar_filas(TABLAS / "memoria_por_modelo.csv") if existe(TABLAS / "memoria_por_modelo.csv") else None
        agrega("escalabilidad", "Análisis de escalabilidad", "computo", "completado" if fe == 96 and fm == 11 else ("en_curso" if compl else ("parcial" if fe else "pendiente")),
               {"hechas": (fe or 0) + (fm or 0), "total": 107}, f"escalabilidad.csv: {fe if fe is not None else 'sin archivo'} de 96 filas; memoria: {fm if fm is not None else 'sin archivo'} de 11")
        sh = [existe(TABLAS / "shap_importancia.csv"), existe(TABLAS / "pesos_por_region_r0_f0.csv")]
        agrega("shap", "Interpretabilidad SHAP", "computo", "completado" if all(sh) else ("en_curso" if compl else ("parcial" if any(sh) else "pendiente")),
               {"hechas": sum(sh), "total": 2}, f"{sum(sh)} de 2 archivos de salida", "Aviso: el script aún no fija semilla, así que hoy no sería reproducible.")

        # documentación y desarrollo: un archivo presente no significa tarea terminada
        manual = {}
        try:
            manual = json.loads((AQUI / "avance_manual.json").read_text(encoding="utf-8")).get("tareas", {})
        except (OSError, ValueError):
            pass
        toc = ""
        try:
            toc = (RAIZ / "_toc.yml").read_text(encoding="utf-8").lower()
        except OSError:
            pass
        def doc(id_, nombre, clase, comprobaciones, nota=None):
            ok = sum(1 for _, v in comprobaciones if v)
            m = manual.get(id_) or {}
            if m.get("completado") and m.get("evidencia"):
                est, ev = "completado", f"Confirmado por el equipo: {m['evidencia']}"
            else:
                est = "por_confirmar" if comprobaciones and ok == len(comprobaciones) else ("parcial" if ok else "sin_evidencia")
                fal = [n for n, v in comprobaciones if not v]
                ev = f"{ok} de {len(comprobaciones)} comprobaciones." + (f" Falta: {', '.join(fal)}." if fal else "")
            agrega(id_, nombre, clase, est, {"hechas": ok, "total": len(comprobaciones)}, ev, nota)

        doc("protocolo12", "Desviaciones registradas en el protocolo", "documentacion",
            [("entrada fechada en la sección 12", seccion_12_con_entradas() is True)])
        doc("libro", "Jupyter Book: capítulos de resultados", "documentacion", [(v, k in toc) for k, v in CAPITULOS.items()],
            "Se busca el nombre del capítulo en _toc.yml. Que exista no prueba que esté terminado.")
        doc("archivos", "Documentación de repositorio", "documentacion",
            [(n, existe(RAIZ / n)) for n in ("MODEL_CARD.md", "DATA_STATEMENT.md", "CITATION.cff", "LICENSE", "references.bib", "artifacts/metadata.json")])
        doc("dash", "Construcción del dashboard predictivo", "desarrollo",
            [("app.py", existe(DASH / "app.py")), ("pages/", existe(DASH / "pages")), ("render.yaml", existe(DASH / "render.yaml")),
             ("model card", existe(DASH / "MODEL_CARD.md")), ("despliegue confirmado", False)],
            "Otro repositorio: solo se comprueba que existan archivos. El despliegue lo confirma el equipo.")
        doc("auditoria", "Auditoría final contra la guía", "documentacion", [("experiments/AUDITORIA_FINAL.md", existe(RAIZ / "experiments" / "AUDITORIA_FINAL.md"))],
            "Se hace cuando exista el resultado final.")

        # costo de las etapas posteriores, con tiempos reales de Telco
        tel = {m: hechos.get(("telco", m), []) for m in MODELOS}
        comb = hechos.get(("telco", "_combinadores"), [])
        if not all(tel.values()) or not comb:
            return salida, (0.0, 0.0, 0.0)
        media = lambda lista, f: sum(f(x) for x in lista) / len(lista)
        seg = lambda x: x["busq"] + x["ajuste"] + x["pred"]
        ajuste_mapem = _mediana([x["trial_med"] for x in comb]) / 3
        robustez = sum(media(tel[m], lambda x: (x["trial_med"] or 0) + x["ajuste"]) for m in MODELOS) * 105 + 1600 * ajuste_mapem
        final_w = sum(media(tel[m], seg) for m in MODELOS) + media(comb, self.seg_tarea)
        pico_final = max(media(tel[m], seg) for m in MODELOS)
        shap_s = 6e6 * sum(sum(x["pred"] for x in tel[m]) / sum(x["n_pred"] for x in tel[m]) for m in MODELOS)
        dash_s = 175 * media(tel["svm"], lambda x: x["ajuste"])
        por_etapa = {"robustez": robustez / n_calc, "tenure": max(final_w / n_calc, pico_final), "holdout": max(final_w / n_calc, pico_final) * 1.25,
                     "shap": shap_s, "comb_dash": dash_s / n_calc}
        centro = 0.0
        for e in salida:
            if e["id"] in por_etapa:
                if e["estado"] == "completado":
                    continue
                e["estim_seg"] = por_etapa[e["id"]]
                centro += por_etapa[e["id"]]
        for e in salida:
            if e["id"] in ("estadistica", "tablas", "escalabilidad") and e["estado"] != "completado":
                e["estim_seg"] = None
        return salida, (0.7 * centro, centro, 1.5 * centro)

    def bucle(self):
        while True:
            try:
                self.ciclo()
            except Exception as ex:   # el monitor nunca debe caerse ni afectar a la corrida
                with self.candado:
                    self.foto = {**self.foto, "error_monitor": repr(ex)[:300]}
            time.sleep(PERIODO)


# ---------------------------------------------------------------------------------- servidor
def crear_manejador(mon, hosts):
    class Manejador(BaseHTTPRequestHandler):
        server_version = "MonitorTelco"
        sys_version = ""

        def log_message(self, *a):
            pass

        def _enviar(self, codigo, cuerpo, tipo):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):
            host = (self.headers.get("Host") or "").split(":")[0]
            if hosts is not None and host not in hosts:
                return self._enviar(403, b"Host no permitido", "text/plain; charset=utf-8")
            ruta = self.path.split("?")[0]
            if ruta in ("/", "/index.html"):
                return self._enviar(200, (AQUI / "index.html").read_bytes(), "text/html; charset=utf-8")
            if ruta == "/api/estado":
                with mon.candado:
                    foto = mon.foto
                return self._enviar(200, json.dumps(foto, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return self._enviar(404, b"No encontrado", "text/plain; charset=utf-8")

    return Manejador


def main():
    ap = argparse.ArgumentParser(description="Monitor de solo lectura de los experimentos")
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--red-local", action="store_true", help="Escucha en toda la red local, sin autenticación, para abrirlo desde el celular")
    a = ap.parse_args()
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex(("127.0.0.1", a.puerto)) == 0:   # en Windows, dos servidores podrían compartir el puerto sin avisar
            print(f"Ya hay un monitor activo en http://127.0.0.1:{a.puerto}. No se abre otro.", flush=True)
            return
    mon = Monitor()
    threading.Thread(target=mon.bucle, daemon=True).start()
    host = "0.0.0.0" if a.red_local else "127.0.0.1"
    hosts = None if a.red_local else {"127.0.0.1", "localhost"}
    srv = ThreadingHTTPServer((host, a.puerto), crear_manejador(mon, hosts))
    print(f"Monitor en http://127.0.0.1:{a.puerto} (Ctrl+C para cerrar). No afecta a los experimentos.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
