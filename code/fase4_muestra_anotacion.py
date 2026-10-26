"""Genera la muestra de anotación PURE-300 y las plantillas de los anotadores.

Muestreo estratificado por documento: asignación proporcional al tamaño, con un
mínimo de min(30, tamaño del documento) por documento y semilla fija.

Salida en datasets/anotacion/:
  - muestra_300_master.csv: con req_id y documento; no se entrega a los anotadores.
  - anotador_1.csv y anotador_2.csv: texto y columnas vacías para es_requisito,
    etiqueta, duda y comentario.
"""
import csv
import os
import random

SEED = 42
TARGET = 300
HERE = os.path.dirname(os.path.abspath(__file__))
HARM = os.path.join(HERE, "..", "datasets", "harmonized")
OUT_DIR = os.path.join(HERE, "..", "datasets", "anotacion")
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
with open(os.path.join(HARM, "corpus_all.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["corpus"] == "PURE":
            rows.append(row)

by_doc = {}
for r in rows:
    by_doc.setdefault(r["group"], []).append(r)

total = len(rows)
docs = sorted(by_doc)
print(f"PURE limpio: {total} requisitos en {len(docs)} documentos")

# Asignación: mínimo garantizado por documento, resto proporcional al tamaño
alloc = {d: min(30, len(by_doc[d])) for d in docs}
remaining = TARGET - sum(alloc.values())
pool_sizes = {d: len(by_doc[d]) - alloc[d] for d in docs}
pool_total = sum(pool_sizes.values())
for d in docs:
    alloc[d] += round(remaining * pool_sizes[d] / pool_total)
# ajuste fino por redondeo
diff = TARGET - sum(alloc.values())
biggest = max(docs, key=lambda d: len(by_doc[d]))
alloc[biggest] += diff

rng = random.Random(SEED)
sample = []
for d in docs:
    chosen = rng.sample(by_doc[d], alloc[d])
    sample.extend(chosen)
    print(f"  {d}: {alloc[d]} de {len(by_doc[d])}")

rng.shuffle(sample)  # orden aleatorio único, idéntico para ambos anotadores
assert len(sample) == TARGET

with open(os.path.join(OUT_DIR, "muestra_300_master.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item", "req_id", "documento", "texto"])
    for i, r in enumerate(sample, 1):
        w.writerow([i, r["req_id"], r["group"], r["text"]])

for annot in (1, 2):
    with open(os.path.join(OUT_DIR, f"anotador_{annot}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "texto", "es_requisito (si/no)", "etiqueta (F o NFR)",
                    "duda (si/no)", "comentario (opcional)"])
        for i, r in enumerate(sample, 1):
            w.writerow([i, r["text"], "", "", "", ""])

print(f"\nGenerado en {OUT_DIR}: muestra_300_master.csv, anotador_1.csv, anotador_2.csv")
