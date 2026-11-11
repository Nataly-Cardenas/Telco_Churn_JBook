# Monitor de experimentos

Tablero local de solo lectura para seguir la corrida confirmatoria.

## Abrirlo

Doble clic en `experiments/monitor/iniciar_monitor.bat`, o desde la raíz del proyecto:

```bash
/c/Users/natal/Downloads/anaconda/envs/ml_project/python.exe experiments/monitor/monitor_servidor.py
```

Luego abre <http://127.0.0.1:8765>. La página se actualiza sola cada 25 s. Abrir `index.html` directamente no sirve: necesita el servidor para leer los datos.

Cerrar el monitor o su ventana no afecta a los experimentos, y se puede volver a abrir en cualquier momento.

## Desde el celular

El servidor solo escucha en este equipo. Para verlo desde el celular en la misma red, ciérralo y ábrelo con `--red-local`, y entra a `http://<IP del equipo>:8765`. Esa opción no tiene contraseña: cualquiera en la red podría ver la página, así que úsala solo en una red de confianza.

## De dónde sale cada dato

| Dato | Fuente | Tipo |
|---|---|---|
| Tareas terminadas | Existencia de `experiments/resultados/crudos/<conjunto>/r<R>_f<F>/<modelo>.pkl` | Verificado |
| Tiempos, evaluaciones y fallos | Campos de cada checkpoint, leídos una sola vez | Verificado |
| Intentos y errores | `experiments/resultados/log_anidado.txt`, leyendo solo lo nuevo | Verificado |
| Procesos, RAM y CPU | `psutil` | Verificado |
| Tareas en curso y desde cuándo | Deducido de la cola FIFO de joblib y de los checkpoints | Derivado |
| Tiempo restante y fin | Tiempos reales por modelo y conjunto; `tiempos_medidos.csv` corregido cuando aún no hay datos | Estimado |
| Evaluaciones de la tarea en curso | Optuna no las registra hasta guardar el checkpoint | No disponible |

## Reglas

- No escribe en el proyecto. Su única escritura es `telco_monitor_cache.json`, en la carpeta temporal del sistema. Borrarla solo hace que vuelva a leer los checkpoints poco a poco.
- Nunca abre los checkpoints del hold-out; solo los cuenta.
- No calcula métricas de pliegue externo. Las pruebas estadísticas solo se cuentan por su columna de control, sin leer p ni efectos.
- El log-loss del bucle interno está oculto por defecto y marcado como provisional.

## Cerrar tareas de documentación

El monitor nunca da por terminada una tarea de documentación o desarrollo solo porque exista un archivo. Para cerrarla, edita `avance_manual.json` con `"completado": true` y una evidencia concreta (commit, URL o ruta). Se refleja en el siguiente ciclo.
