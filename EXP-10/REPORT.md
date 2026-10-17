# Brain Tumour Detection from MRI using Deep Learning

Deep learning mini project: binary classification of brain MRI slices (**Tumour / No Tumour**) with a comparison of a baseline CNN against ImageNet transfer-learning models, explainability with Grad-CAM, and a web demo.

> Research prototype for coursework. Not a medical device and not validated for clinical use.

## 1. Problem and motivation

Brain tumours are diagnosed from MRI, and reading scans is slow and depends on specialist availability. An automated first-pass classifier can prioritise scans for review. The task here is image classification: given a 2-D MRI slice, predict whether a tumour is present. In screening, a **missed tumour (false negative) is the costly error**, so sensitivity is reported alongside accuracy.

Objectives
1. Build a baseline model and compare it with stronger architectures.
2. Evaluate honestly (no data leakage) with metrics suited to a medical task.
3. Explain predictions with Grad-CAM.
4. Deliver a working demo (FastAPI backend + React frontend + PDF report).

## 2. Related work

- **CNNs for medical imaging.** Deep learning has become the standard approach for medical image analysis (Litjens et al., 2017).
- **Transfer learning for brain MRI.** Deepak and Ameer (2019) classified brain tumour MRI using features from a pretrained GoogLeNet, showing that ImageNet transfer works well even with small medical datasets. Transfer learning in general: Pan and Yang (2010).
- **Efficient architectures.** MobileNetV2 (Sandler et al., 2018) uses inverted residuals and depthwise-separable convolutions for small, fast models. EfficientNet (Tan and Le, 2019) scales depth, width and resolution jointly. Both are used here as pretrained backbones.
- **Explainability.** Grad-CAM (Selvaraju et al., 2017) highlights the image regions that drive a prediction, which is important for trust in medical AI.
- **Recent work (2024-2025).** Pretrained CNNs are still the main approach: Disci et al. (2025) compared six fine-tuned models, and Venkanna et al. (2025) used ResNet50 for the same binary Tumour / No Tumour task as this project. Newer work adds Vision Transformers, either alone or combined with CNNs or GRUs (Ahmed et al., 2024; Chandraprabha et al., 2025). Many of these papers report 96-99% accuracy on public Kaggle datasets, which supports this project's warning that very high scores on these datasets may be partly due to easy or overlapping data.
- **Benchmarks.** BraTS (Menze et al., 2015) is the standard multimodal brain-tumour segmentation benchmark; this project uses a simpler slice-level classification dataset.

Full references are in section 9.

## 3. Dataset

About 3,600 labelled MRI images in two folders per split (`dataset/Training`, `dataset/Testing`), classes *No Tumor* and *Yes Tumor*. The dataset was obtained from Kaggle and appears to be derived from the "Brain Tumor MRI Dataset" (Nickparvar), which itself combines figshare, SARTAJ and Br35H images [8]. Its file naming (`Te-gl`, `Te-me`, `Te-pi`, `Te-no`) indicates the original glioma / meningioma / pituitary / no-tumour classes, which were merged into a binary Tumour / No Tumour label for this project.

| | Raw | After cleaning |
|---|---|---|
| Training | 2,916 | 2,743 (2,331 train + 412 validation, stratified) |
| Testing | 705 | 608 |

### Data-quality findings (important)
1. **Exact duplicates.** 173 duplicate files inside Training, and **72 Testing images that are byte-identical to Training images**. All were removed (MD5 hashing) so test results are not inflated.
2. **Near-duplicates.** After exact-duplicate removal, a perceptual-hash (difference hash) check found that **about 46% of the remaining test images (280 of 608) have a visually near-identical image in Training** (Hamming distance 12 or less of 256). These are almost all *No Tumor* images. Only 328 test images are truly distinct from training: 299 tumour and 29 no-tumour (`ml/eval_strict.py`).

Consequence: headline test scores are optimistic, and the no-tumour class is under-represented in the strict subset. Both results are reported below.

Images vary in size (e.g. 236x236, 630x630) and are resized to 128x128 (baseline) or 224x224 (transfer models) and scaled to 0-1.

## 4. Methodology

Framework: TensorFlow / Keras. Training notebook: `ml/brain_tumour_colab.ipynb` (run on Colab T4 GPU). Local script with the same logic: `ml/train_compare.py`.

| Model | Description | Parameters |
|---|---|---|
| Baseline CNN | 3 conv + max-pool blocks, dense 128, dropout 0.5, trained from scratch | 3.30 M |
| MobileNetV2 | ImageNet backbone + global pooling + dropout 0.3 + sigmoid | 2.26 M |
| EfficientNetB0 | ImageNet backbone + global pooling + dropout 0.3 + sigmoid | 4.05 M |

- **Augmentation:** horizontal flip, rotation (+/-5%), zoom (10%), applied only during training.
- **Transfer models:** two stages: (1) train the new head with the backbone frozen (lr 1e-3, up to 15 epochs); (2) fine-tune the top 30 backbone layers with BatchNorm frozen (lr 1e-5, up to 8 epochs).
- **Optimiser / loss:** Adam, binary cross-entropy. Early stopping on validation loss (patience 4), best weights restored.
- **Model selection** uses the validation split only; the test set is used once for final numbers.
- **Metrics:** accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, confusion matrix, at threshold 0.5.
- **Explainability:** Grad-CAM on the final convolutional feature map (`backend/services/inference.py`).

## 5. Results

### 5.1 Cleaned test set (608 images: 299 tumour, 309 no tumour)

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 | AUC | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 0.9951 | 1.0000 | 0.9900 | 1.0000 | 0.9950 | 1.0000 | 3 | 0 |
| **MobileNetV2** | **1.0000*** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| EfficientNetB0 | 0.9951 | 0.9933 | 0.9967 | 0.9935 | 0.9950 | 0.9999 | 1 | 2 |

\* 0 errors out of 608 images. This is not a claim of perfect accuracy: with 608 images, 0 errors only shows that the true accuracy on this kind of data is at least about 99.4% (95% Wilson lower bound), and this test set overlaps with the training data (section 3).

### 5.2 Strict test subset (328 images with no near-duplicate in training: 299 tumour, 29 no tumour)

| Model | Accuracy | Sensitivity | Specificity | F1 | FN | FP |
|---|---|---|---|---|---|---|
| Baseline CNN | 0.9909 | 0.9900 | 1.0000 | 0.9950 | 3 | 0 |
| **MobileNetV2** | **1.0000*** | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| EfficientNetB0 | 0.9970 | 0.9967 | 1.0000 | 0.9983 | 1 | 0 |

\* 0 errors out of 328 images (95% Wilson lower bound on accuracy: about 98.8%), and only 29 of these are no-tumour scans.

### 5.3 External check on images from other sources

The models were also tried on a handful of brain MRI images taken from the web and from papers (not from the Kaggle dataset). **The models classified the tumour images among them as "No Tumor"** (predicted tumour probability 0.00-0.40), while all dataset tumour images were classified correctly. The Grad-CAM heatmap for one such image highlighted the image corners instead of the lesion. This shows the models have learned the characteristics of this one dataset (crop, contrast, borders) and **do not generalise to scans from other sources**. The near-perfect scores above therefore describe performance on this dataset only, not real-world accuracy. This was an informal check on a few images, not a systematic external evaluation.

Figures: `ml/results/roc.png`, `cm_*.png` (confusion matrices), `curves_*.png` (training curves), `misclassified.png`, `gradcam_examples.png`. Raw numbers: `ml/results/metrics.json`, `summary.csv`.

Training cost (Colab T4): baseline 44 s, MobileNetV2 150 s, EfficientNetB0 206 s.

## 6. Discussion

- **Transfer learning helps, but the gap is small.** MobileNetV2 made no errors on the test set; the from-scratch baseline missed 3 of 299 tumours. MobileNetV2 is also smaller than the other two models, so it is the best accuracy/size trade-off and was chosen for deployment.
- **Sensitivity is well supported.** The 299 tumour images in the strict subset are distinct from training, and the models still find 99-100% of them. That is the strongest evidence that the models generalise rather than memorise.
- **Specificity is weakly supported.** Almost every no-tumour test image has a near-duplicate in training, leaving only 29 distinct negatives. "100% specificity" should not be over-claimed.
- **Perfect scores are a warning, not a success.** A 100% result on a public dataset usually means the dataset is easy and/or contaminated, and the external check in section 5.3 confirms the models do not transfer to other sources. Differences of a few images between models (0-3 errors) are within noise and not statistically meaningful.
- **Grad-CAM** is used as a sanity check on whether the model attends to the lesion rather than to the background; see `gradcam_examples.png`.

### Challenges
- Duplicate and overlapping images between train and test (found and handled as above).
- Variable image sizes and quality; an image-quality check (blur, contrast, noise) is included in the app.
- Keras version mismatch between the Colab-saved models and the local environment (fixed by removing an empty `quantization_config` field from the saved model configs).
- An earlier version of the demo displayed hardcoded "classical vs quantum" metrics that were not measured. These were removed; the app now shows only the measured numbers from `metrics.json`.

## 7. Limitations and future work

Limitations
- 2-D slices only; no patient-level split (slices from the same patient may appear in both splits), which can inflate results.
- Binary label only; tumour type (glioma / meningioma / pituitary) is not predicted.
- Single dataset; the informal external check (section 5.3) showed failures on images from other sources, and no systematic external validation was done; no calibration analysis, single training run per model (no confidence intervals).
- No clinical validation. Outputs must not be used for diagnosis.

Future work
1. Patient-level splits and de-duplication with perceptual hashing from the start; k-fold cross-validation with confidence intervals.
2. External validation on a different MRI dataset (other scanners and hospitals).
3. Multi-class tumour typing and tumour segmentation (e.g. U-Net) on BraTS.
4. Vision Transformers and ensemble models; probability calibration and threshold tuning to maximise sensitivity.
5. Hybrid quantum-classical classifier (e.g. PennyLane) as a genuine, separately evaluated experiment; it is not implemented in this project.

## 8. How to reproduce / run

**Train and evaluate (Colab):** open `ml/brain_tumour_colab.ipynb`, select a T4 GPU, put `dataset/` in Google Drive (`MyDrive/brain_tumour/dataset`), run all cells. Copy the outputs into `ml/results/` and `model/`.

**Strict-subset evaluation (local):**
```
backend\venv\Scripts\python.exe -I ml\eval_strict.py
```

**Run the app (two terminals):**
```
cd backend
venv\Scripts\python.exe main.py        # API on http://localhost:8000
cd frontend
npm install
npm run dev                            # UI on the printed local URL
```
The backend loads the model named by `deployed_model` in `ml/results/metrics.json` (currently `mobilenetv2`).

## 9. References

1. Litjens, G., et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60-88.
2. Deepak, S., and Ameer, P. M. (2019). Brain tumor classification using deep CNN features via transfer learning. *Computers in Biology and Medicine*, 111, 103345.
3. Pan, S. J., and Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering*, 22(10), 1345-1359.
4. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
5. Tan, M., and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
7. Menze, B. H., et al. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). *IEEE Transactions on Medical Imaging*, 34(10), 1993-2024.
8. Nickparvar, M. *Brain Tumor MRI Dataset*. Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset (combines figshare, SARTAJ and Br35H images; glioma, meningioma, pituitary and no-tumour classes merged here into a binary Tumour / No Tumour label).
9. Disci, R., Gurcan, F., and Soylu, A. (2025). Advanced brain tumor classification in MR images using transfer learning and pre-trained deep CNN models. *Cancers*, 17(1).
10. Ahmed, M. M., et al. (2024). Brain tumor detection and classification in MRI using hybrid ViT and GRU model with explainable AI in Southern Bangladesh. *Scientific Reports*, 14. https://doi.org/10.1038/s41598-024-71893-3
11. Chandraprabha, K., Ganesan, L., and Baskaran, K. (2025). A novel approach for the detection of brain tumor and its classification via end-to-end vision transformer-CNN architecture. *Frontiers in Oncology*, 15. https://doi.org/10.3389/fonc.2025.1508451
12. Venkanna, G., et al. (2025). Enhanced brain tumor diagnosis using transfer learning medical imaging. *Proc. ICCSCE 2025*, Atlantis Press.
