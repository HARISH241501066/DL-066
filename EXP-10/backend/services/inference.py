import os
import json
import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
METRICS_PATH = os.path.join(PROJECT_ROOT, "ml", "results", "metrics.json")
LEGACY_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "braintumor_binary.h5")


def _load_metrics():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _resolve_model_path():
    """Use the model named `deployed_model` in ml/results/metrics.json, else the original .h5 model."""
    metrics = _load_metrics()
    name = metrics.get("deployed_model") if metrics else None
    if name:
        path = os.path.join(PROJECT_ROOT, "model", f"{name}.keras")
        if os.path.exists(path):
            return name, path
    return "original_model", LEGACY_MODEL_PATH


MODEL_NAME, MODEL_PATH = _resolve_model_path()

# Load model globally to avoid reloading on every request
model = None
input_shape = (224, 224) # Default fallback

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        input_shape = (model.input_shape[1], model.input_shape[2])
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")


METRIC_KEYS = ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc")


def build_evaluation_summary():
    """Real, measured test-set metrics from ml/train_compare.py (None if the script has not been run)."""
    metrics = _load_metrics()
    if not metrics:
        return None
    rows = []
    for name, m in metrics.get("models", {}).items():
        rows.append({"model": name, "deployed": name == MODEL_NAME,
                     **{k: (round(m[k], 4) if k in m else None) for k in METRIC_KEYS},
                     "params": m.get("params")})
    return {"test_images": metrics.get("data", {}).get("test_clean"), "models": rows}


def preprocess_image(image_path: str, target_size: tuple):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    
    # EfficientNet expects unnormalized inputs usually if using keras.applications,
    # but standard practice for custom trained models might be 1/255.
    # We will just normalize to 0-1 for safety unless it's handled inside the model.
    img_array = img_resized / 255.0
    return np.expand_dims(img_array, axis=0), img_rgb

def _gradcam_cam(image_array):
    """Grad-CAM for the tumour logit. Handles flat CNNs and models with a nested pretrained backbone."""
    x = tf.constant(image_array)
    backbone = next((l for l in model.layers if isinstance(l, tf.keras.Model) and not isinstance(l, tf.keras.Sequential)), None)
    if backbone is not None:
        # transfer-learning layout: [input, augment, rescaling, backbone, pooling, dropout, dense]
        i = model.layers.index(backbone)
        pre, head = model.layers[i - 1], model.layers[i + 1:]
        with tf.GradientTape() as tape:
            feats = backbone(pre(x), training=False)
            tape.watch(feats)
            z = feats
            for layer in head:
                z = layer(z, training=False)
            score = z[:, 0]
    else:
        last_conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
        grad_model = tf.keras.Model(model.inputs, [last_conv.output, model.output])
        with tf.GradientTape() as tape:
            feats, out = grad_model(x, training=False)
            score = out[:, 0]
    grads = tape.gradient(score, feats)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * feats, axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)


def generate_gradcam(image_array, original_image, image_path):
    if model is None:
        return None

    try:
        cam = _gradcam_cam(image_array)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0])), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

        output_filename = "heatmap_" + os.path.basename(image_path)
        output_path = os.path.join(os.path.dirname(image_path), output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return output_filename
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

def process_and_infer(image_path: str):
    if model is None:
        # Mock response if model is not loaded (for testing without model)
        return generate_mock_results()
        
    img_array, original_image = preprocess_image(image_path, input_shape)
    
    prediction = model.predict(img_array)[0][0]
    is_tumor = prediction > 0.5
    
    # Generate Heatmap
    heatmap_filename = generate_gradcam(img_array, original_image, image_path)
    
    probability = float(prediction) if is_tumor else float(1.0 - prediction)
    
    confidence = "High" if probability > 0.90 else "Intermediate" if probability > 0.70 else "Low"
    risk_level = "HIGH RISK INDICATOR" if is_tumor else "LOW RISK INDICATOR"
    
    if not is_tumor and probability > 0.7:
        risk_level = "LOW RISK INDICATOR"
    if 0.4 <= prediction <= 0.6:
        risk_level = "UNCERTAIN — SPECIALIST REVIEW ADVISED"

    return {
        "prediction": "Tumor Detected" if is_tumor else "No Tumor Detected",
        "probability": f"{probability * 100:.1f}%",
        "confidence": confidence,
        "risk_level": risk_level,
        "heatmap_image": heatmap_filename,
        "model_name": MODEL_NAME,
        "evaluation": build_evaluation_summary(),
    }

def generate_mock_results():
    return {
        "prediction": "Tumor Detected",
        "probability": "94.7%",
        "confidence": "High",
        "risk_level": "HIGH RISK INDICATOR",
        "heatmap_image": None,
        "model_name": "demo (model not loaded)",
        "evaluation": build_evaluation_summary(),
    }
