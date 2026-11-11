"""Sella el análisis confirmatorio antes de abrir el hold-out: guarda el SHA-256 de cada tabla y la hora.

Si el sello ya existe no lo reescribe, para que no pueda rehacerse después de ver el hold-out.
Uso: python experiments/confirmatorio/sellar_pre_holdout.py
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import RESULTADOS  # noqa: E402

SELLO = RESULTADOS / "sello_pre_holdout.json"
if SELLO.exists():
    sys.exit(f"ya existe {SELLO}; no se reescribe")
if (RESULTADOS / "crudos" / "telco_holdout").exists():
    sys.exit("el hold-out ya tiene checkpoints: el sello no sería previo")
tablas = RESULTADOS / "tablas"
sello = {"fecha": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "nota": "análisis confirmatorio sobre los 4 conjuntos, antes de evaluar el hold-out",
         "archivos": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tablas.iterdir()) if p.is_file()}}
SELLO.write_text(json.dumps(sello, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(sello, indent=1, ensure_ascii=False))
