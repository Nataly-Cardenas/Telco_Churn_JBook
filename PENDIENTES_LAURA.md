# Pendientes para Laura

Lista de lo que falta antes de la entrega. Este archivo no aparece en el libro.

## 1. Comprobar la revisión de repositorios (capítulo 4, tablas 4.6 y 4.7)

La hizo Claude el 2026-11-10 y una persona debe confirmar que es correcta. (HECHO)

- Abrir cada repositorio de `experiments/literatura/repositorios_revision.csv` y comprobar que la descripción y la diferencia sean ciertas. Empezar por `Hariharan6880/Saas-churn-prediction` (`scripts/step6c_regime_ensemble.py`), que es el más cercano.  (HECHO)

- Hojear `experiments/literatura/repositorios_resultados.csv` por si se pasó alguno relevante. 

- `experiments/literatura/busqueda_repositorios.py` repite la búsqueda; los conteos pueden cambiar con el tiempo.

- Si algo cambia, corregir el CSV y regenerar con `python experiments/confirmatorio/construir_capitulos.py 04`. Si aparece un repositorio con el mismo mecanismo, hay que revisar también la declaración de novedad.

## 2. Repetir la búsqueda antes de entregar
La guía pide repetirla y registrar la fecha. La última en Web of Science y Scopus fue el 2026-11-03.

- Volver a correr las cadenas de la tabla 4.4 con el acceso de Uninorte.
- Agregar las filas nuevas al Sheet, descargarlo como Excel y correr `python experiments/literatura/exportar_sheet.py ruta/al/archivo.xlsx`.
- Regenerar el capítulo con `python experiments/confirmatorio/construir_capitulos.py 04`.

## 3. Leer a texto completo los dos artículos de 2010 y 2012 (recomendado)

Son los antecedentes más cercanos y hoy el capítulo 4 se apoya solo en sus resúmenes. Confirmar si los pesos cambian por subárea o son globales, y ajustar la tabla 4.3 si hace falta. (Avance 2026-11-13: ya están identificados y citados como Xia 2010 y Xia 2012; el resumen del de 2010 confirma pesos por subárea. Faltan, si se puede, volumen, páginas y coautores, y leer el de 2012. No es obligatorio para entregar.) (Intentado el 2026-11-13: el texto completo de ambos artículos no es accesible; el de 2010 solo con resumen)

## 4. Leer la declaración de uso de IA (capítulo 10)

Dice "Firmado por el equipo" con los nombres de las dos. Debe estar de acuerdo con lo que afirma.

## 5. Paper

Revisar las secciones 5 a 7 de `paper.md`, que tienen un borrador con los resultados reales; subir el PDF final y actualizar lo que se refiera al paper en el libro. (Avance 2026-11-13: secciones 5 a 7 de `paper.md` revisadas contra el manuscrito y sin diferencias en las cifras; aviso de estado actualizado; manuscrito compilado como `paper/paper_regionalmapem.pdf` y enlazado en la página del paper. Falta: pasar el manuscrito a la plantilla sn-jnl, resolver las marcas rojas de financiación, herramientas y DOI de Zenodo, y los DOI de `csiszar1984` y `romano2006`.)

## 6. DOI de Zenodo (de último)

Cuando el libro esté cerrado: activar el repositorio en Zenodo, crear una release en GitHub y citar el DOI en el capítulo 10.
