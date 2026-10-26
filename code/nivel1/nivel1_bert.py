"""Nivel 1, BERT: ambos modos y ambos corpus, con 5 semillas.

Bloques: con fuga en A y B (StratifiedKFold de 10 particiones, semilla de
partición 42) y sin fuga en A y B (LeaveOneGroupOut por proyecto: 15 y 34
iteraciones). Total: 345 entrenamientos con la configuración de NoRBERT descrita
en el artículo.

Reanudable por iteración: cada (bloque, semilla, iteración) terminado se agrega
de inmediato al CSV de predicciones de su bloque; al relanzar, los terminados se
detectan y se saltan. Interrumpir con Ctrl+C es seguro.

Salidas (datasets/nivel1/): con-fuga/nivel1_confuga_{A,B}_predicciones_bert.csv
y sin-fuga-logo/nivel1_logo_{A,B}_predicciones_bert.csv, con columnas req_id,
etiqueta, prediccion, iteracion, proyecto, semilla.
"""
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

SPLIT_SEED = 42
SEMILLAS = [13, 42, 77, 101, 2026]
MODEL_NAME = "bert-base-cased"
EPOCHS = 16
LR = 2e-5
BATCH = 16
MAX_LEN = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE, flush=True)

CSV_CORPUS = os.path.join("..", "..", "datasets", "harmonized", "corpus_all.csv")
DIR_NIVEL1 = os.path.join("..", "..", "datasets", "nivel1")
CORPUS = {"A": "PROMISE", "B": "PROMISE_exp_new"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
todo = pd.read_csv(CSV_CORPUS, dtype=str)
datos = {}
for letra, nombre in CORPUS.items():
    d = todo[todo.corpus == nombre]
    datos[letra] = (d.text.values, d.label_binary.values, d.group.values,
                    d.req_id.values)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def encode(texts, labels, label2id):
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                    padding="max_length", return_tensors="pt")
    y = torch.tensor([label2id[l] for l in labels])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


def entrena_y_predice(X_tr, y_tr, X_te, seed):
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
    for _ in range(EPOCHS):
        for ids, mask, yy in dl_tr:
            ids, mask, yy = ids.to(DEVICE), mask.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=yy)
            out.loss.backward()
            opt.step(); sched.step()
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


def ruta_bloque(modo, letra):
    sub = "con-fuga" if modo == "confuga" else "sin-fuga-logo"
    return os.path.join(DIR_NIVEL1, sub,
                        f"nivel1_{modo}_{letra}_predicciones_bert.csv")


def completados(ruta):
    if not os.path.exists(ruta):
        return set()
    hechos = pd.read_csv(ruta, dtype=str)
    return set(zip(hechos.semilla.astype(int), hechos.iteracion.astype(int)))


def anexa(ruta, filas):
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["req_id", "etiqueta", "prediccion",
                                          "iteracion", "proyecto", "semilla"])
        if nuevo:
            w.writeheader()
        w.writerows(filas)


bloques = []
for letra in ("A", "B"):
    X, y, g, rid = datos[letra]
    skf = list(StratifiedKFold(n_splits=10, shuffle=True,
                               random_state=SPLIT_SEED).split(X, y))
    logo = list(LeaveOneGroupOut().split(X, y, g))
    bloques.append(("confuga", letra, skf))
    bloques.append(("logo", letra, logo))

total = sum(len(splits) for _, _, splits in bloques) * len(SEMILLAS)
print(f"tanda total: {total} entrenamientos", flush=True)

hechos_global = 0
for modo, letra, splits in bloques:
    X, y, g, rid = datos[letra]
    ruta = ruta_bloque(modo, letra)
    ya = completados(ruta)
    for seed in SEMILLAS:
        for k, (tr, te) in enumerate(splits):
            hechos_global += 1
            if (seed, k) in ya:
                continue
            pred = entrena_y_predice(X[tr], y[tr], X[te], seed)
            anexa(ruta, [{"req_id": rid[i], "etiqueta": y[i],
                          "prediccion": p, "iteracion": k,
                          "proyecto": g[i], "semilla": seed}
                         for i, p in zip(te, pred)])
            print(f"[{hechos_global}/{total}] {modo}/{letra} semilla {seed} "
                  f"iteracion {k} lista", flush=True)

print("tanda completa")
