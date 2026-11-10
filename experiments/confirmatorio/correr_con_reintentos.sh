#!/usr/bin/env bash
# Relanza la validación anidada si el sistema se queda sin recursos. Las tareas guardadas no se repiten
# y cada tarea es determinista por sus semillas, así que un reinicio no cambia los resultados.
PY=${PY:-python}
N=${1:-10}
for intento in $(seq 1 60); do
  echo "=== intento $intento $(date) con $N procesos" >> experiments/resultados/log_anidado.txt
  if PYTHONIOENCODING=utf-8 "$PY" experiments/confirmatorio/correr_anidado.py todos "$N" >> experiments/resultados/log_anidado.txt 2>&1; then
    echo "fin correcto"; exit 0
  fi
  sleep 60
done
echo "fin con errores"; exit 1
