# Brain Tumour Detection from MRI (Deep Learning Mini Project)

Binary classification of brain MRI slices (**Tumour / No Tumour**). Compares a baseline CNN with ImageNet transfer-learning models (MobileNetV2, EfficientNetB0), explains predictions with Grad-CAM, and ships a web demo (FastAPI + React) that produces a PDF report.

> Research prototype for coursework. Not a medical device; do not use for diagnosis.

Full write-up (related work, methodology, results, limitations, references): **[REPORT.md](REPORT.md)**

## Results (cleaned test set, 608 images)

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---|---|---|---|---|
| Baseline CNN | 99.5% | 99.0% | 100% | 0.995 | 1.000 |
| **MobileNetV2 (deployed)** | **100%*** | 100% | 100% | 1.000 | 1.000 |
| EfficientNetB0 | 99.5% | 99.7% | 99.4% | 0.995 | 0.9999 |

\* 0 errors on 608 images, so the true accuracy on this kind of data is at least about 99.4% (95% lower bound). It is **not** a claim of perfect or real-world accuracy.

**Read this before quoting the numbers.** The dataset contains duplicates: 72 test images were byte-identical to training images (removed), and about 46% of the remaining test images have a near-identical twin in training. Scores are therefore optimistic. On a strict subset with no near-duplicates (328 images: 299 tumour, 29 no tumour) the models still reach 99-100% sensitivity, but specificity rests on only 29 images. In an informal check, tumour images taken from the web and papers (outside the Kaggle dataset) were classified as "No Tumor", so the model does **not** generalise to scans from other sources. Details in the report (sections 3, 5.3).

## Not included in this repository

To keep the repository small, the following are not committed: `dataset/` (download it from Kaggle, see Dataset below and place it in `dataset/Training` and `dataset/Testing`), the non-deployed models `baseline_cnn.keras` and `efficientnetb0.keras` (re-create them with `ml/brain_tumour_colab.ipynb`), the Python virtual environment, `node_modules`, and uploaded files. Only the deployed model `model/mobilenetv2.keras` is included.

## Project structure

```
backend/    FastAPI API: inference, Grad-CAM, image-quality check, PDF report
frontend/   React (Vite) web UI
ml/         Training and evaluation
  brain_tumour_colab.ipynb   Colab notebook (train, evaluate, Grad-CAM)
  train_compare.py           Same training pipeline as a local script
  eval_strict.py             Re-evaluation on the near-duplicate-free test subset
  results/                   metrics.json, summary.csv, ROC / confusion matrix / curves / Grad-CAM plots
model/      Trained models (.keras) + original braintumor_binary.h5 (fallback)
dataset/    Training/ and Testing/ (No Tumor, Yes Tumor), from Kaggle
REPORT.md   Project report
```

## Setup and run the app

Requirements: Python 3.10, Node.js 18+.

**Backend** (Windows):
```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
API runs at http://localhost:8000 (`GET /health`, `POST /analyze` with an MRI image upload).

**Frontend** (second terminal):
```
cd frontend
npm install
npm run dev
```
Open the URL printed by Vite, upload an MRI image (JPG / PNG), and view the prediction, Grad-CAM heatmap, image quality, model comparison table and downloadable PDF report.

The backend loads the model named by `deployed_model` in `ml/results/metrics.json` (currently `mobilenetv2`, file `model/mobilenetv2.keras`). If that is missing it falls back to `model/braintumor_binary.h5`. Use TensorFlow 2.21 / Keras 3.12 as pinned in your venv; models saved by a newer Keras may need the empty `quantization_config` field removed from their config to load.

## Reproduce the training

**On Colab (recommended, GPU):**
1. Upload `ml/brain_tumour_colab.ipynb` to Colab and select Runtime > T4 GPU.
2. Put the `dataset/` folder in Google Drive at `MyDrive/brain_tumour/dataset` (or upload `dataset.zip` and set `USE_ZIP = True`).
3. Run all cells. Outputs are saved to `MyDrive/brain_tumour/results/`.
4. Copy `metrics.json` and the plots into `ml/results/`, the `.keras` models into `model/`, and set `"deployed_model": "<name>"` in `metrics.json`.

**Locally (CPU, slower):**
```
backend\venv\Scripts\pip install scikit-learn pandas
backend\venv\Scripts\python.exe -I ml\train_compare.py --epochs 10 --ft-epochs 5
backend\venv\Scripts\python.exe -I ml\eval_strict.py
```

## Method summary

- **Data cleaning:** MD5 hashing removes exact duplicates and train/test overlap; stratified 85/15 train/validation split; perceptual hashing used to build the strict test subset.
- **Models:** baseline 3-block CNN (128x128, from scratch); MobileNetV2 and EfficientNetB0 (224x224, ImageNet weights, head training then fine-tuning of the top 30 layers).
- **Training:** Adam, binary cross-entropy, augmentation (flip / rotation / zoom), early stopping on validation loss.
- **Evaluation:** accuracy, precision, sensitivity (recall), specificity, F1, ROC-AUC, confusion matrices, on a test set never used for model selection.
- **Explainability:** Grad-CAM heatmaps in the API and the notebook.

## Limitations

2-D slices without patient-level splits, single dataset, no external validation, no clinical validation. See Section 7 of the report.

## Dataset

Kaggle "Brain Tumor MRI Dataset" by Masoud Nickparvar (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which combines figshare, SARTAJ and Br35H images. The original glioma / meningioma / pituitary / no-tumour classes are merged into a binary Tumour / No Tumour label here.
