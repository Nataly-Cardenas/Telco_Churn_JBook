#!/usr/bin/env bash
# Etapas que siguen a la validación anidada, en el orden del protocolo.
# Cada paso reanuda desde sus puntos de control si se cae; se reintenta hasta 3 veces.
# Con "final" como segundo argumento sigue con el hold-out (exige el sello previo), el análisis final,
# la escalabilidad (1 proceso, para no contaminar los tiempos) y SHAP.
# Uso: bash experiments/confirmatorio/correr_etapas_post.sh [n_procesos] [final]
cd "$(dirname "$0")/../.." || exit 1
PY=/c/Users/natal/Downloads/anaconda/envs/ml_project/python.exe
N=${1:-10}
LOG=experiments/resultados/log_post.txt
export PYTHONIOENCODING=utf-8
PASOS=("complementos_confirmatorio.py dashboard $N" "robustez.py $N" "correr_finales.py tenure $N")
if [ "$2" = "final" ]; then
  if [ ! -f experiments/resultados/sello_pre_holdout.json ]; then echo "falta el sello previo al hold-out" >> "$LOG"; exit 1; fi
  PASOS+=("correr_finales.py holdout $N" "analizar.py" "complementos_confirmatorio.py escalabilidad 1" "complementos_confirmatorio.py shap $N")
fi
for paso in "${PASOS[@]}"; do
  for intento in 1 2 3; do
    echo "=== $(date) $paso (intento $intento)" >> "$LOG"
    if $PY experiments/confirmatorio/$paso >> "$LOG" 2>&1; then break; fi
    if [ "$intento" = 3 ]; then echo "=== $(date) fallo definitivo: $paso" >> "$LOG"; exit 1; fi
    sleep 60
  done
done
echo "=== $(date) fin de etapas post" >> "$LOG"
