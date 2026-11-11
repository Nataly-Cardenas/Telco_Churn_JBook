@echo off
rem Abre el monitor de solo lectura. Cerrar esta ventana no afecta a los experimentos.
start "" http://127.0.0.1:8765
"C:\Users\natal\Downloads\anaconda\envs\ml_project\python.exe" "%~dp0monitor_servidor.py" --puerto 8765
pause
