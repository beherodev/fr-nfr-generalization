"""Nivel 2, BERT: unión A+B (con y sin fuga) y cruces, con 5 semillas.

Bloques: union_confuga (StratifiedKFold de 10 particiones sobre los 973
requisitos, semilla 42), union_logo (LeaveOneGroupOut por los 49 proyectos),
cruce_AaB (entrena con A completo y evalúa B completo) y cruce_BaA (al revés).
Total: 305 entrenamientos con la configuración de NoRBERT descrita en el
artículo.

Reanudable por iteración, igual que el nivel 1. Interrumpir con Ctrl+C es seguro.

Salidas en datasets/nivel2/: union-con-fuga/, union-sin-fuga-logo/ y cruces/.
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
DIR_NIVEL2 = os.path.join("..", "..", "datasets", "nivel2")
SUBDIR = {"union_confuga": "union-con-fuga", "union_logo": "union-sin-fuga-logo",
          "cruce_AaB": "cruces", "cruce_BaA": "cruces"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
todo = pd.read_csv(CSV_CORPUS, dtype=str)
datos = {}
for letra, nombre in (("A", "PROMISE"), ("B", "PROMISE_exp_new")):
    d = todo[todo.corpus == nombre]
    datos[letra] = (d.text.values, d.label_binary.values, d.group.values,
                    d.req_id.values)
datos["AB"] = tuple(np.concatenate([datos["A"][i], datos["B"][i]])
                    for i in range(4))


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


def ruta_cond(cond):
    return os.path.join(DIR_NIVEL2, SUBDIR[cond],
                        f"nivel2_{cond}_predicciones_bert.csv")


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


X_AB, y_AB, g_AB, rid_AB = datos["AB"]
skf = list(StratifiedKFold(n_splits=10, shuffle=True,
                           random_state=SPLIT_SEED).split(X_AB, y_AB))
logo = list(LeaveOneGroupOut().split(X_AB, y_AB, g_AB))
bloques = [("union_confuga", skf), ("union_logo", logo),
           ("cruce_AaB", None), ("cruce_BaA", None)]

total = (len(skf) + len(logo) + 2) * len(SEMILLAS)
print(f"tanda total: {total} entrenamientos", flush=True)

hechos_global = 0
for cond, splits in bloques:
    ruta = ruta_cond(cond)
    ya = completados(ruta)
    for seed in SEMILLAS:
        if cond.startswith("union"):
            for k, (tr, te) in enumerate(splits):
                hechos_global += 1
                if (seed, k) in ya:
                    continue
                pred = entrena_y_predice(X_AB[tr], y_AB[tr], X_AB[te], seed)
                anexa(ruta, [{"req_id": rid_AB[i], "etiqueta": y_AB[i],
                              "prediccion": p, "iteracion": k,
                              "proyecto": g_AB[i], "semilla": seed}
                             for i, p in zip(te, pred)])
                print(f"[{hechos_global}/{total}] {cond} semilla {seed} "
                      f"iteracion {k} lista", flush=True)
        else:
            hechos_global += 1
            if (seed, 0) in ya:
                continue
            origen, destino = ("A", "B") if cond == "cruce_AaB" else ("B", "A")
            Xo, yo, _, _ = datos[origen]
            Xd, yd, gd, ridd = datos[destino]
            pred = entrena_y_predice(Xo, yo, Xd, seed)
            anexa(ruta, [{"req_id": ridd[i], "etiqueta": yd[i],
                          "prediccion": p, "iteracion": 0,
                          "proyecto": gd[i], "semilla": seed}
                         for i, p in enumerate(pred)])
            print(f"[{hechos_global}/{total}] {cond} semilla {seed} lista",
                  flush=True)

print("tanda completa")
