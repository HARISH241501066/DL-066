"""Re-evaluate the trained models on a *strict* test subset with no near-duplicates of any training image.

Exact-duplicate removal (done in training) is not enough: many Testing images are re-saved / lightly altered copies of Training
images. Here a Testing image is kept only if its 256-bit difference hash is more than HAMMING_MIN bits away from every
Training image. Results are added to ml/results/metrics.json under `strict` for each model.

Usage (project root):  backend\\venv\\Scripts\\python.exe -I ml\\eval_strict.py
"""
import hashlib
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset")
METRICS = os.path.join(ROOT, "ml", "results", "metrics.json")
HAMMING_MIN = 12
SIZES = {"baseline_cnn": 128, "mobilenetv2": 224, "efficientnetb0": 224}


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def dhash(p):
    im = cv2.resize(cv2.imread(p, 0), (17, 16), interpolation=cv2.INTER_AREA)
    return (im[:, 1:] > im[:, :-1]).flatten()


def rows(split):
    out = []
    for label, cls in enumerate(("No Tumor", "Yes Tumor")):
        folder = os.path.join(DATA, split, cls)
        for fn in sorted(os.listdir(folder)):
            p = os.path.join(folder, fn)
            out.append((p, label, md5(p), dhash(p)))
    return out


train, test = rows("Training"), rows("Testing")
train_md5 = {r[2] for r in train}
seen, clean = set(), []
for r in test:  # same exact-duplicate cleaning as training
    if r[2] in train_md5 or r[2] in seen:
        continue
    seen.add(r[2]); clean.append(r)
A = np.array([r[3] for r in train])
dist = np.array([(A != r[3]).sum(1).min() for r in clean])
strict = [r for r, d in zip(clean, dist) if d > HAMMING_MIN]
y = np.array([r[1] for r in strict])
print(f"clean test {len(clean)} -> strict test {len(strict)} (tumour {int(y.sum())}, no tumour {int((1 - y).sum())})")


def load(paths, size):
    return np.stack([cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (size, size)) / 255.0 for p in paths]).astype("float32")


out = json.load(open(METRICS))
out["strict_test"] = {"hamming_min": HAMMING_MIN, "n": len(strict), "n_tumour": int(y.sum()), "n_no_tumour": int((1 - y).sum()),
                      "n_clean_test": len(clean)}
for name, size in SIZES.items():
    model = tf.keras.models.load_model(os.path.join(ROOT, "model", f"{name}.keras"), compile=False)
    prob = model.predict(load([r[0] for r in strict], size), verbose=0).ravel()
    pred = (prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    m = {"accuracy": float((tp + tn) / len(y)), "precision": float(precision_score(y, pred, zero_division=0)),
         "recall_sensitivity": float(recall_score(y, pred)), "specificity": float(tn / (tn + fp)),
         "f1": float(f1_score(y, pred)), "roc_auc": float(roc_auc_score(y, prob)),
         "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
    out["models"][name]["strict"] = m
    print(name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, m["confusion_matrix"])
    tf.keras.backend.clear_session()
json.dump(out, open(METRICS, "w"), indent=2)
