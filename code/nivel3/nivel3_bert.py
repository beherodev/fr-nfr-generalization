"""Nivel 3, BERT: transferencia hacia PURE con 5 semillas.

Tres condiciones, todas sin particiones: entrenar con la unión A+B completa
(973 requisitos), solo con A (625) o solo con B (348), y predecir los 958
requisitos de PURE. PURE nunca se usa para entrenar. Total: 15 entrenamientos.

A aporta 625 requisitos de 15 proyectos y B aporta 348 de 34, de modo que
cantidad y diversidad apuntan en direcciones opuestas y la comparación las separa.

Reanudable: cada (condición, semilla) terminada se agrega de inmediato a su CSV;
al relanzar se detecta y se salta. Interrumpir con Ctrl+C es seguro.

Salidas en datasets/nivel3/: transferencia-union/, transferencia-soloA/ y
transferencia-soloB/.
"""
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

SEMILLAS = [13, 42, 77, 101, 2026]
MODEL_NAME = "bert-base-cased"
EPOCHS = 16
LR = 2e-5
BATCH = 16
MAX_LEN = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE, flush=True)

DATOS = os.path.join("..", "..", "datasets")
CSV_CORPUS = os.path.join(DATOS, "harmonized", "corpus_all.csv")
CSV_GOLD = os.path.join(DATOS, "anotacion", "gold_pure300.csv")
DIR_NIVEL3 = os.path.join(DATOS, "nivel3")
SUBDIR = {"union": "transferencia-union", "soloA": "transferencia-soloA",
          "soloB": "transferencia-soloB"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
todo = pd.read_csv(CSV_CORPUS, dtype=str)
entren = {}
for letra, nombre in (("A", "PROMISE"), ("B", "PROMISE_exp_new")):
    d = todo[todo.corpus == nombre]
    entren[letra] = (d.text.values, d.label_binary.values)
entren["AB"] = tuple(np.concatenate([entren["A"][i], entren["B"][i]])
                     for i in range(2))
pure = todo[todo.corpus == "PURE"].reset_index(drop=True)
gold = pd.read_csv(CSV_GOLD, dtype=str)
gold_por_id = dict(zip(gold.req_id, gold.etiqueta))
esreq_por_id = dict(zip(gold.req_id, gold.es_requisito))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def encode(texts, labels, label2id):
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                    padding="max_length", return_tensors="pt")
    y = torch.tensor([label2id[l] for l in labels])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


def entrena_y_predice(X_tr, y_tr, X_te, seed, avance=""):
    set_seed(seed)
    labels_sorted = sorted(set(y_tr))
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels_sorted)).to(DEVICE)
    dl_tr = DataLoader(encode(X_tr, y_tr, label2id), batch_size=BATCH,
                       shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = len(dl_tr) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    for ep in range(EPOCHS):
        for ids, mask, yy in dl_tr:
            ids, mask, yy = ids.to(DEVICE), mask.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=yy)
            out.loss.backward()
            opt.step(); sched.step()
        print(f"    {avance} epoca {ep + 1}/{EPOCHS}", flush=True)
    model.eval()
    enc = tokenizer(list(X_te), truncation=True, max_length=MAX_LEN,
                    padding="max_length", return_tensors="pt")
    dl_te = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"]),
                       batch_size=64)
    preds = []
    with torch.no_grad():
        for ids, mask in dl_te:
            logits = model(input_ids=ids.to(DEVICE),
                           attention_mask=mask.to(DEVICE)).logits
            preds.extend(logits.argmax(-1).cpu().numpy())
    id2label = {i: l for l, i in label2id.items()}
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return [id2label[p] for p in preds]


def ruta_cond(cond):
    return os.path.join(DIR_NIVEL3, SUBDIR[cond],
                        f"nivel3_{cond}_predicciones_bert.csv")


def completados(ruta):
    if not os.path.exists(ruta):
        return set()
    hechos = pd.read_csv(ruta, dtype=str)
    return set(hechos.semilla.astype(int))


def anexa(ruta, filas):
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["req_id", "documento", "modelo",
                                          "prediccion", "etiqueta_gold",
                                          "es_requisito", "semilla"])
        if nuevo:
            w.writeheader()
        w.writerows(filas)


total = 3 * len(SEMILLAS)
print(f"tanda total: {total} entrenamientos", flush=True)

hechos_global = 0
for cond, clave in (("union", "AB"), ("soloA", "A"), ("soloB", "B")):
    ruta = ruta_cond(cond)
    ya = completados(ruta)
    faltan = [s for s in SEMILLAS if s not in ya]
    print(f"{cond}: {len(ya)} semillas ya hechas, faltan {len(faltan)}",
          flush=True)
    X_tr, y_tr = entren[clave]
    for seed in SEMILLAS:
        hechos_global += 1
        if seed in ya:
            continue
        marca = f"[{hechos_global}/{total}] {cond} semilla {seed}"
        print(f"{marca}: entrenando con {len(y_tr)} requisitos", flush=True)
        pred = entrena_y_predice(X_tr, y_tr, pure.text.values, seed, marca)
        anexa(ruta, [{"req_id": pure.req_id[i], "documento": pure.group[i],
                      "modelo": "BERT (NoRBERT)", "prediccion": pred[i],
                      "etiqueta_gold": gold_por_id.get(pure.req_id[i], ""),
                      "es_requisito": esreq_por_id.get(pure.req_id[i], ""),
                      "semilla": seed}
                     for i in range(len(pure))])
        print(f"[{hechos_global}/{total}] {cond} semilla {seed} lista",
              flush=True)

print("tanda completa")
