"""Genera las Tablas 1, 2 y 3 del artículo a partir de las predicciones guardadas.

No entrena ningún modelo: lee las predicciones de datasets/nivel1, nivel2 y nivel3
y calcula el F1 macro (media del F1 de F y de NFR).

  - Tabla 1: F1 macro sobre todas las predicciones de cada condición.
  - Tabla 2: media del F1 macro por proyecto, solo proyectos con ambas clases.
  - Tabla 3: media del F1 macro por documento de PURE-300, sobre los 265
    requisitos confirmados (es_requisito = si).

Los modelos clásicos son deterministas y tienen un solo valor. BERT se reporta
como media y desviación estándar poblacional sobre sus cinco semillas.

Al final se compara cada celda, redondeada a dos decimales, con la del artículo.
"""
import csv
import os
import statistics

DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
CLASICOS = ["TF-IDF + MNB", "TF-IDF + LinearSVC", "fastText + LogReg"]
COLUMNAS = ["Bayes ingenuo", "SVM lineal", "fastText", "BERT"]


def leer(*ruta):
    with open(os.path.join(DATOS, *ruta), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f1_macro(pares):
    """pares: lista de (etiqueta real, predicción) con clases F y NFR."""
    f1s = []
    for clase in ("F", "NFR"):
        tp = sum(1 for v, p in pares if v == clase and p == clase)
        fp = sum(1 for v, p in pares if v != clase and p == clase)
        fn = sum(1 for v, p in pares if v == clase and p != clase)
        precision = tp / (tp + fp) if tp + fp else 0.0
        exhaustividad = tp / (tp + fn) if tp + fn else 0.0
        suma = precision + exhaustividad
        f1s.append(2 * precision * exhaustividad / suma if suma else 0.0)
    return sum(f1s) / 2


def f1_global(filas):
    return f1_macro([(r["etiqueta"], r["prediccion"]) for r in filas])


def f1_por_grupo(filas, grupo):
    """Media del F1 macro de los grupos que contienen ambas clases."""
    grupos = {}
    for r in filas:
        grupos.setdefault(r[grupo], []).append((r["etiqueta"], r["prediccion"]))
    valores = [f1_macro(p) for p in grupos.values() if len({v for v, _ in p}) == 2]
    return statistics.fmean(valores)


def fila(clasicos, bert, metrica):
    """Una fila de tabla: un valor por modelo clásico y (media, desviación) de BERT."""
    valores = [metrica([r for r in clasicos if r["modelo"] == m]) for m in CLASICOS]
    semillas = sorted({r["semilla"] for r in bert})
    por_semilla = [metrica([r for r in bert if r["semilla"] == s]) for s in semillas]
    return valores + [(statistics.fmean(por_semilla), statistics.pstdev(por_semilla))]


def predicciones(nivel, carpeta, prefijo):
    return (leer(nivel, carpeta, f"{prefijo}_predicciones_clasicos.csv"),
            leer(nivel, carpeta, f"{prefijo}_predicciones_bert.csv"))


def tabla_1():
    condiciones = [
        ("PROMISE con fuga", "nivel1", "con-fuga", "nivel1_confuga_A"),
        ("PROMISE sin fuga", "nivel1", "sin-fuga-logo", "nivel1_logo_A"),
        ("PROMISE_exp con fuga", "nivel1", "con-fuga", "nivel1_confuga_B"),
        ("PROMISE_exp sin fuga", "nivel1", "sin-fuga-logo", "nivel1_logo_B"),
        ("PROMISE + PROMISE_exp con fuga", "nivel2", "union-con-fuga", "nivel2_union_confuga"),
        ("PROMISE + PROMISE_exp sin fuga", "nivel2", "union-sin-fuga-logo", "nivel2_union_logo"),
    ]
    return [(nombre, fila(*predicciones(nivel, carpeta, prefijo), f1_global))
            for nombre, nivel, carpeta, prefijo in condiciones]


def tabla_2():
    por_proyecto = lambda filas: f1_por_grupo(filas, "proyecto")
    condiciones = [
        ("Cruce PROMISE -> PROMISE_exp", "nivel2", "cruces", "nivel2_cruce_AaB"),
        ("PROMISE_exp sin fuga", "nivel1", "sin-fuga-logo", "nivel1_logo_B"),
        ("Cruce PROMISE_exp -> PROMISE", "nivel2", "cruces", "nivel2_cruce_BaA"),
        ("PROMISE sin fuga", "nivel1", "sin-fuga-logo", "nivel1_logo_A"),
    ]
    return [(nombre, fila(*predicciones(nivel, carpeta, prefijo), por_proyecto))
            for nombre, nivel, carpeta, prefijo in condiciones]


def tabla_3():
    estandar = {r["req_id"]: r for r in leer("anotacion", "gold_pure300.csv")}
    confirmados = {k for k, r in estandar.items() if r["es_requisito"] == "si"}

    def con_etiqueta(filas):
        # La etiqueta real sale del estándar PURE-300; solo cuentan los confirmados.
        return [dict(r, etiqueta=estandar[r["req_id"]]["etiqueta"])
                for r in filas if r["req_id"] in confirmados]

    por_documento = lambda filas: f1_por_grupo(filas, "documento")
    condiciones = [
        ("PROMISE + PROMISE_exp", "transferencia-union", "nivel3_union"),
        ("PROMISE", "transferencia-soloA", "nivel3_soloA"),
        ("PROMISE_exp", "transferencia-soloB", "nivel3_soloB"),
    ]
    tabla = []
    for nombre, carpeta, prefijo in condiciones:
        clasicos, bert = predicciones("nivel3", carpeta, prefijo)
        tabla.append((nombre, fila(con_etiqueta(clasicos), con_etiqueta(bert), por_documento)))
    return tabla


# Valores publicados en el artículo: tres clásicos, media de BERT y desviación de BERT.
PUBLICADO = {
    "Tabla 1": [
        [0.89, 0.91, 0.84, 0.91, 0.01],
        [0.51, 0.55, 0.61, 0.72, 0.03],
        [0.87, 0.90, 0.84, 0.91, 0.01],
        [0.77, 0.77, 0.77, 0.86, 0.02],
        [0.89, 0.89, 0.82, 0.92, 0.00],
        [0.74, 0.73, 0.72, 0.82, 0.01],
    ],
    "Tabla 2": [
        [0.52, 0.53, 0.61, 0.67, 0.04],
        [0.69, 0.65, 0.62, 0.69, 0.03],
        [0.70, 0.63, 0.65, 0.74, 0.02],
        [0.48, 0.51, 0.59, 0.68, 0.03],
    ],
    "Tabla 3": [
        [0.48, 0.48, 0.42, 0.53, 0.02],
        [0.36, 0.40, 0.42, 0.57, 0.06],
        [0.44, 0.45, 0.50, 0.49, 0.02],
    ],
}


def formato_bert(media, desviacion):
    desv = "<0.01" if round(desviacion, 2) == 0 else f"{desviacion:.2f}"
    return f"{media:.2f} ± {desv}"


def imprimir(titulo, tabla):
    ancho = max(len(nombre) for nombre, _ in tabla)
    print(f"\n{titulo}")
    print(" " * ancho + "".join(f"{c:>15}" for c in COLUMNAS))
    for nombre, valores in tabla:
        celdas = [f"{v:.2f}" for v in valores[:3]] + [formato_bert(*valores[3])]
        print(f"{nombre:<{ancho}}" + "".join(f"{c:>15}" for c in celdas))


def comparar(titulo, tabla):
    diferencias = []
    for (nombre, valores), publicado in zip(tabla, PUBLICADO[titulo]):
        calculado = valores[:3] + list(valores[3])
        for columna, c, p in zip(COLUMNAS + ["desv. BERT"], calculado, publicado):
            if round(c, 2) != p:
                diferencias.append(f"{titulo}, {nombre}, {columna}: "
                                   f"calculado {c:.4f}, publicado {p:.2f}")
    return diferencias


def main():
    tablas = {"Tabla 1": tabla_1(), "Tabla 2": tabla_2(), "Tabla 3": tabla_3()}
    diferencias = []
    for titulo, tabla in tablas.items():
        imprimir(titulo, tabla)
        diferencias += comparar(titulo, tabla)

    total = sum(len(v) * 5 for v in PUBLICADO.values())
    print()
    if diferencias:
        print(f"{len(diferencias)} de {total} celdas no coinciden con el artículo:")
        for d in diferencias:
            print(f"  {d}")
    else:
        print(f"Las {total} celdas coinciden con el artículo.")


if __name__ == "__main__":
    main()
