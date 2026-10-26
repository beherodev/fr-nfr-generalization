# Evaluación de clasificadores de requisitos en tres niveles

Código y datos para reproducir los resultados del artículo:

> B. Hernández Rojas, V. Rodríguez López y C. A. Fernández y Fernández. *Una evaluación comparativa del desempeño de clasificadores de requisitos funcionales*. JIISIC 2026, Lecture Notes in Computer Science, Springer.

El estudio evalúa cuatro clasificadores de requisitos funcionales (F) y no funcionales (NFR) en tres niveles que alejan progresivamente el conjunto de prueba del de entrenamiento:

1. **Nivel 1**: dentro del mismo corpus, con fuga (validación cruzada aleatoria) y sin fuga (dejando fuera un proyecto completo en cada iteración).
2. **Nivel 2**: entre corpus (PROMISE y PROMISE_exp).
3. **Nivel 3**: hacia documentos industriales (PURE-300).

![Protocolo de evaluación en tres niveles](docs/protocolo.png)

*Protocolo de evaluación. Cada nivel aleja el conjunto de prueba del de entrenamiento.*

Los clasificadores son Bayes ingenuo y SVM lineal sobre TF-IDF, regresión logística sobre vectores fastText, y BERT con la configuración de NoRBERT.

## Comprobación rápida

Las predicciones de todos los modelos están incluidas. Para regenerar las Tablas 1, 2 y 3 del artículo sin entrenar nada:

```
cd code
python tablas.py
```

El script imprime las tres tablas y compara cada celda con el valor publicado. Solo usa la biblioteca estándar de Python.

## Estructura

```
code/
  fase0_prepare_corpora.ipynb    corpus armonizado (datasets/harmonized/corpus_all.csv)
  fase4_muestra_anotacion.py     muestra PURE-300 y plantillas de anotación
  fase4_acuerdo.ipynb            acuerdo entre anotadores y estándar PURE-300
  nivel1/                        nivel 1: clásicos (notebook) y BERT (script)
  nivel2/                        nivel 2: unión A+B y cruces
  nivel3/                        nivel 3: transferencia hacia PURE-300
  tablas.py                      Tablas 1, 2 y 3 del artículo
  environment.yml                entorno de conda con las versiones usadas
docs/
  protocolo.png                  figura del protocolo de evaluación
datasets/
  norbert-replication/.../promise_nfr.csv   PROMISE (fuente original)
  promise_exp.csv                           PROMISE_exp (fuente original)
  pure-xml/XMLZIPFile/                      seis documentos de PURE (fuente original)
  harmonized/corpus_all.csv                 los tres corpus limpios y armonizados
  anotacion/                                muestra, anotaciones, adjudicación y estándar PURE-300
  nivel1/, nivel2/, nivel3/                 predicciones y métricas de cada nivel
```

## Entorno

```
conda env create -f code/environment.yml
conda activate crossreq
```

Los experimentos se ejecutaron en macOS con Apple M4 Pro y 24 GB de memoria; BERT se ajustó sobre el acelerador MPS. Los vectores `fasttext-wiki-news-subwords-300` (cerca de 1 GB) se descargan automáticamente con gensim la primera vez, y `bert-base-cased` se descarga desde Hugging Face.

## Orden de ejecución

Cada paso lee lo que produjo el anterior. Todos los scripts y notebooks se ejecutan desde su propia carpeta.

| Paso | Archivo | Produce |
|---|---|---|
| 1 | `code/fase0_prepare_corpora.ipynb` | `corpus_all.csv`; verifica su SHA-256 |
| 2 | `code/fase4_muestra_anotacion.py` | muestra PURE-300 y plantillas de los anotadores |
| 3 | `code/fase4_acuerdo.ipynb` | kappa de Cohen, tabla de adjudicación y `gold_pure300.csv` |
| 4 | `code/nivel1/nivel1_bert.py`, `nivel1_clasicos.ipynb` | predicciones y métricas del nivel 1 |
| 5 | `code/nivel2/nivel2_bert.py`, `nivel2.ipynb` | predicciones y métricas del nivel 2 |
| 6 | `code/nivel3/nivel3_bert.py`, `nivel3.ipynb` | predicciones y métricas del nivel 3 |
| 7 | `code/tablas.py` | Tablas 1, 2 y 3 |

Los pasos 2 y 3 no hace falta repetirlos: dependen de las respuestas de los anotadores y de la adjudicación, que se incluyen. Correrlos regenera los mismos archivos.

Los notebooks de los clásicos reutilizan las predicciones guardadas si existen; para recalcularlas basta con borrar los archivos `*_predicciones_clasicos.csv`. Los modelos clásicos son deterministas. BERT suma 665 entrenamientos (cinco semillas: 13, 42, 77, 101 y 2026) y tarda varias horas; sus scripts se pueden interrumpir y reanudar.

Las métricas de los notebooks de nivel se calculan sobre todas las predicciones de cada condición. Las Tablas 2 y 3 del artículo reportan, en cambio, la media del F1 macro por proyecto y por documento; ambas las calcula `tablas.py`.

## Datos

### Fuentes externas

| Corpus | Fuente | Licencia |
|---|---|---|
| PROMISE | Cleland-Huang et al. (2007). Se usa `promise_nfr.csv` del paquete de NoRBERT, Hey et al. (2020), [doi:10.5281/zenodo.3833661](https://doi.org/10.5281/zenodo.3833661). El conjunto original está en [doi:10.5281/zenodo.268542](https://doi.org/10.5281/zenodo.268542) | Paquete de NoRBERT: GPL-3.0. Conjunto original: CC BY 4.0 |
| PROMISE_exp | Lima et al. (2019), *Software Engineering Repositories: Expanding the PROMISE Database*, SBES. Archivo tomado de [github.com/AleksandarMitrevski/se-requirements-classification](https://github.com/AleksandarMitrevski/se-requirements-classification) | Sin licencia declarada en la fuente |
| PURE | Ferrari, Spagnolo y Gnesi (2017), [doi:10.5281/zenodo.1414117](https://doi.org/10.5281/zenodo.1414117). Se incluyen los seis documentos que traen requisitos marcados | CC BY 4.0 |

Estos archivos se redistribuyen sin modificaciones, con fines de reproducibilidad, y conservan la licencia y la autoría de su fuente. Si alguno de los autores prefiere que su archivo no se redistribuya aquí, se retirará y se dejará solo el enlace.

### Datos de este trabajo

- `harmonized/corpus_all.csv`: 1931 requisitos. PROMISE (625: 255 F y 370 NFR, 15 proyectos), los 348 requisitos de PROMISE_exp que no están en PROMISE (191 F y 157 NFR, 34 proyectos) y 958 requisitos de PURE sin etiqueta.
- `anotacion/`: la muestra PURE-300, las respuestas de los dos anotadores, la adjudicación de los 95 desacuerdos con su justificación, el manual que recibieron los anotadores y el estándar `gold_pure300.csv` (300 textos, 265 confirmados como requisitos: 166 F y 99 NFR).
- `nivel1/`, `nivel2/`, `nivel3/`: las predicciones de cada modelo (con su iteración, proyecto o documento y semilla) y las métricas.

## Licencia

El código se distribuye bajo la licencia MIT y los datos de este trabajo bajo CC BY 4.0; ver `LICENSE`. Los datos de fuentes externas conservan su licencia original.
