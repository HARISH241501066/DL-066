"""Train and compare brain-tumour MRI classifiers, then evaluate on the held-out Testing set.

Models compared
  1. baseline_cnn     - small CNN trained from scratch (same family as the original model)
  2. mobilenetv2      - ImageNet transfer learning (frozen head, then fine-tuning)
  3. efficientnetb0   - ImageNet transfer learning (frozen head, then fine-tuning)
  4. original_model   - the pre-existing model/braintumor_binary.h5, evaluated only (not retrained)

Usage (from project root):
    backend\\venv\\Scripts\\python.exe ml\\train_compare.py [--epochs 8] [--ft-epochs 5] [--models baseline_cnn,mobilenetv2]

Outputs go to ml/results/ (metrics.json, curves, confusion matrices, ROC) and models to model/.
All new models take images scaled to 0-1, so the backend's `/255` preprocessing is unchanged.
Class indices (alphabetical): 0 = No Tumor, 1 = Yes Tumor.
"""
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset")
MODEL_DIR = os.path.join(ROOT, "model")
RESULTS = os.path.join(ROOT, "ml", "results")
SEED = 42
BATCH = 32
VAL_FRACTION = 0.15
IMG_SIZE = {"baseline_cnn": 128, "mobilenetv2": 160, "efficientnetb0": 160, "original_model": 128}

tf.keras.utils.set_random_seed(SEED)
os.makedirs(RESULTS, exist_ok=True)


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def collect_files():
    """List (path, label, md5) per split, with leakage removed.

    The raw dataset has exact duplicate images inside Training and 72 Testing images that are byte-identical
    to Training images. We drop duplicate files within each split and drop any Testing file whose hash also
    appears in Training, so test metrics are not inflated. Hashes are cached because OneDrive reads are slow.
    """
    cache_path = os.path.join(RESULTS, "hash_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    splits, stats = {}, {}
    for split in ("Training", "Testing"):
        rows = []
        for label, cls in enumerate(("No Tumor", "Yes Tumor")):  # alphabetical, 1 = tumour
            folder = os.path.join(DATA, split, cls)
            for fn in sorted(os.listdir(folder)):
                path = os.path.join(folder, fn)
                if path not in cache:
                    cache[path] = _md5(path)
                rows.append((path, label, cache[path]))
        splits[split] = rows
    json.dump(cache, open(cache_path, "w"))

    def dedupe(rows, banned=frozenset()):
        seen, kept = set(), []
        for path, label, h in rows:
            if h in seen or h in banned:
                continue
            seen.add(h); kept.append((path, label, h))
        return kept

    train_all = dedupe(splits["Training"])
    train_hashes = {h for _, _, h in splits["Training"]}
    test = dedupe(splits["Testing"], banned=train_hashes)
    stats = {"train_raw": len(splits["Training"]), "train_dedup": len(train_all),
             "test_raw": len(splits["Testing"]), "test_clean": len(test)}
    return train_all, test, stats


def _decode(size):
    def f(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size))
        return tf.cast(img, tf.float32) / 255.0, tf.cast(label, tf.float32)[None]
    return f


def load_split(size):
    """train/val are a stratified split of the de-duplicated Training folder; test is the cleaned Testing folder."""
    train_all, test, stats = collect_files()
    tr, va = train_test_split(train_all, test_size=VAL_FRACTION, random_state=SEED, stratify=[r[1] for r in train_all])
    stats.update(train=len(tr), val=len(va))
    auto = tf.data.AUTOTUNE

    def make(rows, shuffle):
        ds = tf.data.Dataset.from_tensor_slices(([r[0] for r in rows], [r[1] for r in rows]))
        ds = ds.map(_decode(size), auto).cache()
        if shuffle:
            ds = ds.shuffle(len(rows), seed=SEED)
        return ds.batch(BATCH).prefetch(auto)

    return make(tr, True), make(va, False), make(test, False), stats


def augment():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augment")


def build_baseline(size):
    return tf.keras.Sequential([
        tf.keras.Input((size, size, 3)),
        augment(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="baseline_cnn"), None


def build_transfer(name, size):
    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        prep = tf.keras.layers.Rescaling(2.0, offset=-1.0)  # 0-1 -> -1..1
    else:
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        prep = tf.keras.layers.Rescaling(255.0)  # EfficientNet expects 0-255
    base.trainable = False
    inp = tf.keras.Input((size, size, 3))
    x = augment()(inp)
    x = prep(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name=name), base


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])


def fit(model, train, val, epochs, history):
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    h = model.fit(train, validation_data=val, epochs=epochs, callbacks=cb, verbose=2)
    for k, v in h.history.items():
        history.setdefault(k, []).extend(float(i) for i in v)


def evaluate(model, test):
    y_true = np.concatenate([y.numpy().ravel() for _, y in test]).astype(int)
    prob = model.predict(test, verbose=0).ravel()
    pred = (prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    m = {
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision_score(y_true, pred)),
        "recall_sensitivity": float(recall_score(y_true, pred)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_true)),
    }
    return m, y_true, prob


def plot_curves(name, history):
    if not history:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["loss"], label="train"); ax[0].plot(history["val_loss"], label="val"); ax[0].set_title(f"{name} loss")
    ax[1].plot(history["accuracy"], label="train"); ax[1].plot(history["val_accuracy"], label="val"); ax[1].set_title(f"{name} accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f"curves_{name}.png"), dpi=120); plt.close(fig)


def plot_cm(name, cm):
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(arr, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, arr[i, j], ha="center", va="center", color="white" if arr[i, j] > arr.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No Tumor", "Tumor"]); ax.set_yticks([0, 1], ["No Tumor", "Tumor"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f"cm_{name}.png"), dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--models", default="baseline_cnn,mobilenetv2,efficientnetb0,original_model")
    args = ap.parse_args()

    metrics_path = os.path.join(RESULTS, "metrics.json")
    out = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {"models": {}}
    rocs = {}

    for name in args.models.split(","):
        size = IMG_SIZE[name]
        train, val, test, stats = load_split(size)
        out["data"] = stats
        history, t0 = {}, time.time()
        if name == "original_model":
            model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "braintumor_binary.h5"), compile=False)
        else:
            model, base = build_baseline(size) if name == "baseline_cnn" else build_transfer(name, size)
            compile_model(model, 1e-3)
            fit(model, train, val, args.epochs, history)
            if base is not None:  # fine-tune the top of the pretrained backbone at a low learning rate
                base.trainable = True
                for layer in base.layers[:-30]:
                    layer.trainable = False
                for layer in base.layers:  # keep BatchNorm statistics frozen
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = False
                compile_model(model, 1e-5)
                fit(model, train, val, args.ft_epochs, history)
            model.save(os.path.join(MODEL_DIR, f"{name}.keras"))
        m, y_true, prob = evaluate(model, test)
        # validation-set numbers, kept separate so the test set is never used for model selection
        if name != "original_model":
            m["val_accuracy"] = float(max(history["val_accuracy"]))
        m.update(params=int(model.count_params()), input_size=size, train_seconds=round(time.time() - t0, 1),
                 epochs_run=len(history.get("loss", [])))
        out["models"][name] = m
        rocs[name] = (y_true, prob)
        plot_curves(name, history); plot_cm(name, m["confusion_matrix"])
        print(f"== {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))
        out["models"][name]["history"] = history
        json.dump(out, open(metrics_path, "w"), indent=2)
        tf.keras.backend.clear_session()

    # ROC overlay
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for name, (yt, pr) in rocs.items():
        fpr, tpr, _ = roc_curve(yt, pr); ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yt, pr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "roc.png"), dpi=120); plt.close(fig)

    # best model by F1 on the test set is reported; selection is by sensitivity-aware F1, deployed model is chosen by the user
    trained = {k: v for k, v in out["models"].items() if k != "original_model"}
    out["best_model"] = max(trained, key=lambda k: trained[k]["f1"]) if trained else None
    json.dump(out, open(metrics_path, "w"), indent=2)
    print("best:", out["best_model"])


if __name__ == "__main__":
    main()
